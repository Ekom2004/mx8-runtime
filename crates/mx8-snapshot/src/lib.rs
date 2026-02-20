#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use mx8_core::dataset_link::DatasetLink;
use mx8_core::types::{ManifestHash, ManifestRecord, MANIFEST_SCHEMA_VERSION};
use mx8_manifest_store::{intent_key_for_base, LockOwner, ManifestStore, ManifestStoreError};
use thiserror::Error;
use tracing::{info, warn};

pub mod labels;
pub mod pack_dir;
pub mod video_stage1;

#[cfg(feature = "s3")]
pub mod pack_s3;

#[derive(Debug, Error)]
pub enum SnapshotError {
    #[error("dataset link parse error: {0}")]
    Link(#[from] mx8_core::dataset_link::DatasetLinkParseError),
    #[error("manifest store error: {0}")]
    Store(#[from] ManifestStoreError),
    #[error("dev manifest path is required to create a new snapshot")]
    DevManifestPathRequired,
    #[error("dev manifest read error: {0}")]
    DevManifestIo(#[from] std::io::Error),
    #[error("dev manifest parse error: {0}")]
    DevManifestParse(String),
    #[error("dev manifest invariant violated: {0}")]
    DevManifestInvariant(String),
    #[error("indexing is not supported for dataset link base: {0}")]
    IndexUnsupported(String),
    #[error("indexing IO error: {0}")]
    IndexIo(String),
    #[error("indexing parse error: {0}")]
    IndexParse(String),
    #[error("s3 indexing requires feature 's3' (dataset base: {0})")]
    S3FeatureRequired(String),
    #[error("s3 indexing failed: {0}")]
    S3Index(String),
    #[error("snapshot wait timed out after {0:?}")]
    WaitTimeout(Duration),
}

#[derive(Debug, Clone)]
pub struct ResolvedSnapshot {
    pub intent_key: Option<String>,
    pub manifest_hash: ManifestHash,
    pub manifest_bytes: Vec<u8>,
    pub manifest_bytes_materialized: bool,
    pub schema_version: u32,
}

#[derive(Debug)]
struct IndexedManifest {
    manifest_hash: ManifestHash,
    manifest_bytes: u64,
    canonical_bytes: Vec<u8>,
    canonical_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct SnapshotResolverConfig {
    pub lock_stale_after: Duration,
    pub wait_timeout: Duration,
    pub poll_interval: Duration,
    pub dev_manifest_path: Option<PathBuf>,
    pub recursive: bool,
    pub materialize_manifest_bytes: bool,
}

impl Default for SnapshotResolverConfig {
    fn default() -> Self {
        Self {
            lock_stale_after: Duration::from_secs(60),
            wait_timeout: Duration::from_secs(30),
            poll_interval: Duration::from_millis(100),
            dev_manifest_path: None,
            recursive: true,
            materialize_manifest_bytes: true,
        }
    }
}

#[derive(Clone)]
pub struct SnapshotResolver {
    store: Arc<dyn ManifestStore>,
    cfg: SnapshotResolverConfig,
}

impl SnapshotResolver {
    pub fn new(store: Arc<dyn ManifestStore>, cfg: SnapshotResolverConfig) -> Self {
        Self { store, cfg }
    }

    pub fn resolve(&self, link: &str, owner: LockOwner) -> Result<ResolvedSnapshot, SnapshotError> {
        let parsed = DatasetLink::parse(link)?;

        match parsed {
            DatasetLink::Pinned { manifest_hash, .. } => {
                let hash = ManifestHash(manifest_hash);
                let bytes = if self.cfg.materialize_manifest_bytes {
                    self.store.get_manifest_bytes(&hash)?
                } else {
                    Vec::new()
                };
                info!(
                    target: "mx8_proof",
                    event = "snapshot_resolved",
                    mode = "pinned",
                    manifest_hash = %hash.0,
                    "resolved pinned snapshot"
                );
                Ok(ResolvedSnapshot {
                    intent_key: None,
                    manifest_hash: hash,
                    manifest_bytes: bytes,
                    manifest_bytes_materialized: self.cfg.materialize_manifest_bytes,
                    schema_version: MANIFEST_SCHEMA_VERSION,
                })
            }
            DatasetLink::Plain(base) => self.resolve_for_intent(&base, false, owner),
            DatasetLink::Refresh(base) => self.resolve_for_intent(&base, true, owner),
        }
    }

    fn resolve_for_intent(
        &self,
        base: &str,
        refresh: bool,
        owner: LockOwner,
    ) -> Result<ResolvedSnapshot, SnapshotError> {
        let intent_key = intent_key_for_base(base);

        if !refresh {
            if let Some(hash) = self.store.get_current_snapshot(&intent_key)? {
                let bytes = if self.cfg.materialize_manifest_bytes {
                    self.store.get_manifest_bytes(&hash)?
                } else {
                    Vec::new()
                };
                info!(
                    target: "mx8_proof",
                    event = "snapshot_resolved",
                    mode = "plain",
                    intent_key = %intent_key,
                    manifest_hash = %hash.0,
                    "resolved current snapshot"
                );
                return Ok(ResolvedSnapshot {
                    intent_key: Some(intent_key),
                    manifest_hash: hash,
                    manifest_bytes: bytes,
                    manifest_bytes_materialized: self.cfg.materialize_manifest_bytes,
                    schema_version: MANIFEST_SCHEMA_VERSION,
                });
            }
        }

        if refresh {
            info!(
                target: "mx8_proof",
                event = "snapshot_refresh_requested",
                intent_key = %intent_key,
                "refresh requested"
            );
        }

        if let Some(guard) =
            self.store
                .try_acquire_index_lock(&intent_key, self.cfg.lock_stale_after, owner)?
        {
            let _guard = guard;
            info!(
                target: "mx8_proof",
                event = "snapshot_indexer_elected",
                intent_key = %intent_key,
                "elected indexer"
            );

            let indexed = match self.cfg.dev_manifest_path.clone() {
                Some(path) => {
                    let (records, canonical_bytes) = load_dev_manifest_tsv(&path)?;
                    indexed_manifest_from_bytes(
                        records,
                        canonical_bytes,
                        self.cfg.materialize_manifest_bytes,
                    )?
                }
                None => index_manifest_for_base(
                    base,
                    self.cfg.recursive,
                    self.cfg.materialize_manifest_bytes,
                )?,
            };
            let manifest_hash = indexed.manifest_hash.clone();
            match indexed.canonical_path.as_ref() {
                Some(path) => self.store.put_manifest_file(&manifest_hash, path)?,
                None => self
                    .store
                    .put_manifest_bytes(&manifest_hash, &indexed.canonical_bytes)?,
            }
            self.store
                .set_current_snapshot(&intent_key, &manifest_hash)?;
            if let Some(path) = indexed.canonical_path.as_ref() {
                let _ = std::fs::remove_file(path);
            }

            info!(
                target: "mx8_proof",
                event = "snapshot_resolved",
                mode = if refresh { "refresh" } else { "create" },
                intent_key = %intent_key,
                manifest_hash = %manifest_hash.0,
                manifest_bytes = indexed.manifest_bytes,
                "created snapshot"
            );
            return Ok(ResolvedSnapshot {
                intent_key: Some(intent_key),
                manifest_hash,
                manifest_bytes: indexed.canonical_bytes,
                manifest_bytes_materialized: self.cfg.materialize_manifest_bytes,
                schema_version: MANIFEST_SCHEMA_VERSION,
            });
        }

        // Not the indexer: wait for current snapshot to appear.
        let start = Instant::now();
        info!(
            target: "mx8_proof",
            event = "snapshot_index_wait",
            intent_key = %intent_key,
            wait_timeout_ms = self.cfg.wait_timeout.as_millis() as u64,
            "waiting for indexer"
        );
        while start.elapsed() < self.cfg.wait_timeout {
            if let Some(hash) = self.store.get_current_snapshot(&intent_key)? {
                let bytes = if self.cfg.materialize_manifest_bytes {
                    self.store.get_manifest_bytes(&hash)?
                } else {
                    Vec::new()
                };
                info!(
                    target: "mx8_proof",
                    event = "snapshot_resolved",
                    mode = "wait",
                    intent_key = %intent_key,
                    manifest_hash = %hash.0,
                    waited_ms = start.elapsed().as_millis() as u64,
                    "observed snapshot"
                );
                return Ok(ResolvedSnapshot {
                    intent_key: Some(intent_key),
                    manifest_hash: hash,
                    manifest_bytes: bytes,
                    manifest_bytes_materialized: self.cfg.materialize_manifest_bytes,
                    schema_version: MANIFEST_SCHEMA_VERSION,
                });
            }
            std::thread::sleep(self.cfg.poll_interval);
        }

        warn!(
            target: "mx8_proof",
            event = "snapshot_index_wait_timeout",
            intent_key = %intent_key,
            waited_ms = start.elapsed().as_millis() as u64,
            "timed out waiting for snapshot"
        );
        Err(SnapshotError::WaitTimeout(self.cfg.wait_timeout))
    }
}

fn index_manifest_for_base(
    base: &str,
    recursive: bool,
    materialize_manifest_bytes: bool,
) -> Result<IndexedManifest, SnapshotError> {
    let trimmed = base.trim();
    if trimmed.starts_with("s3://") {
        #[cfg(feature = "s3")]
        {
            return index_s3_prefix(trimmed, recursive, materialize_manifest_bytes);
        }
        #[cfg(not(feature = "s3"))]
        {
            return Err(SnapshotError::S3FeatureRequired(trimmed.to_string()));
        }
    }

    // Local FS prefix indexing (directory path).
    if std::path::Path::new(trimmed).is_dir() {
        let (records, canonical_bytes) = index_local_prefix(trimmed, recursive)?;
        return indexed_manifest_from_bytes(records, canonical_bytes, materialize_manifest_bytes);
    }

    Err(SnapshotError::IndexUnsupported(trimmed.to_string()))
}

fn indexed_manifest_from_bytes(
    _records: Vec<ManifestRecord>,
    canonical_bytes: Vec<u8>,
    materialize_manifest_bytes: bool,
) -> Result<IndexedManifest, SnapshotError> {
    let manifest_hash = ManifestHash(mx8_manifest_store::sha256_hex(&canonical_bytes));
    let manifest_bytes = canonical_bytes.len() as u64;
    if materialize_manifest_bytes {
        return Ok(IndexedManifest {
            manifest_hash,
            manifest_bytes,
            canonical_bytes,
            canonical_path: None,
        });
    }

    let path = new_manifest_temp_path();
    std::fs::write(&path, &canonical_bytes)
        .map_err(|e| SnapshotError::IndexIo(format!("write temp manifest file failed: {e}")))?;
    Ok(IndexedManifest {
        manifest_hash,
        manifest_bytes,
        canonical_bytes: Vec::new(),
        canonical_path: Some(path),
    })
}

fn new_manifest_temp_path() -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    path.push(format!("mx8-manifest-{}-{nanos}.tsv", std::process::id()));
    path
}

#[cfg(feature = "s3")]
struct CanonicalManifestSink {
    materialize_manifest_bytes: bool,
    canonical_bytes: Vec<u8>,
    file_path: Option<PathBuf>,
    writer: Option<std::io::BufWriter<std::fs::File>>,
    hasher: mx8_manifest_store::Sha256Hasher,
    bytes_written: u64,
}

#[cfg(feature = "s3")]
struct CanonicalManifestSinkOutput {
    manifest_hash: ManifestHash,
    manifest_bytes: u64,
    canonical_bytes: Vec<u8>,
    canonical_path: Option<PathBuf>,
}

#[cfg(feature = "s3")]
impl CanonicalManifestSink {
    fn new(materialize_manifest_bytes: bool) -> Result<Self, SnapshotError> {
        let (file_path, writer) = if materialize_manifest_bytes {
            (None, None)
        } else {
            let path = new_manifest_temp_path();
            let file = std::fs::File::create(&path)
                .map_err(|e| SnapshotError::IndexIo(format!("create temp manifest failed: {e}")))?;
            (Some(path), Some(std::io::BufWriter::new(file)))
        };

        let mut sink = Self {
            materialize_manifest_bytes,
            canonical_bytes: Vec::new(),
            file_path,
            writer,
            hasher: mx8_manifest_store::sha256_streaming_new(),
            bytes_written: 0,
        };
        let header = format!("schema_version={MANIFEST_SCHEMA_VERSION}\n");
        sink.write_bytes(header.as_bytes())?;
        Ok(sink)
    }

    fn append_record(&mut self, record: &ManifestRecord) -> Result<(), SnapshotError> {
        let line = manifest_record_tsv_line(record);
        self.write_bytes(line.as_bytes())
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), SnapshotError> {
        self.hasher.update(bytes);
        self.bytes_written = self.bytes_written.saturating_add(bytes.len() as u64);
        if self.materialize_manifest_bytes {
            self.canonical_bytes.extend_from_slice(bytes);
            return Ok(());
        }
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| SnapshotError::IndexIo("missing manifest writer".to_string()))?;
        use std::io::Write as _;
        writer
            .write_all(bytes)
            .map_err(|e| SnapshotError::IndexIo(format!("write temp manifest failed: {e}")))?;
        Ok(())
    }

    fn finish(mut self) -> Result<CanonicalManifestSinkOutput, SnapshotError> {
        if let Some(writer) = self.writer.as_mut() {
            use std::io::Write as _;
            writer
                .flush()
                .map_err(|e| SnapshotError::IndexIo(format!("flush temp manifest failed: {e}")))?;
        }
        let digest = self.hasher.finalize();
        Ok(CanonicalManifestSinkOutput {
            manifest_hash: ManifestHash(mx8_manifest_store::sha256_to_lower_hex(&digest)),
            manifest_bytes: self.bytes_written,
            canonical_bytes: self.canonical_bytes,
            canonical_path: self.file_path,
        })
    }
}

fn parse_canonical_manifest_tsv_bytes(
    bytes: &[u8],
) -> Result<(Vec<ManifestRecord>, Vec<u8>), SnapshotError> {
    let s = std::str::from_utf8(bytes)
        .map_err(|e| SnapshotError::IndexParse(format!("manifest is not utf-8: {e}")))?;

    let mut lines = s.lines();
    let first = lines
        .by_ref()
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| SnapshotError::IndexParse("empty manifest".to_string()))?;

    let Some((k, v)) = first.split_once('=') else {
        return Err(SnapshotError::IndexParse(
            "manifest header missing schema_version".to_string(),
        ));
    };
    if k.trim() != "schema_version" {
        return Err(SnapshotError::IndexParse(
            "manifest header must be schema_version=<n>".to_string(),
        ));
    }
    let schema_version: u32 = v
        .trim()
        .parse()
        .map_err(|_| SnapshotError::IndexParse("invalid schema_version".to_string()))?;
    if schema_version != MANIFEST_SCHEMA_VERSION {
        return Err(SnapshotError::IndexParse(format!(
            "unsupported schema_version {schema_version}"
        )));
    }

    let mut records: Vec<ManifestRecord> = Vec::new();
    for (line_no, raw) in lines.enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 5 {
            return Err(SnapshotError::IndexParse(format!(
                "line {}: expected 5 columns (sample_id<TAB>location<TAB>byte_offset<TAB>byte_length<TAB>decode_hint)",
                line_no + 2
            )));
        }
        let sample_id: u64 = cols[0].trim().parse().map_err(|_| {
            SnapshotError::IndexParse(format!("line {}: bad sample_id", line_no + 2))
        })?;
        let location = cols[1].trim().to_string();
        if location.contains('\n') || location.contains('\t') {
            return Err(SnapshotError::IndexParse(format!(
                "line {}: location must not contain tabs/newlines",
                line_no + 2
            )));
        }
        let byte_offset = if cols[2].trim().is_empty() {
            None
        } else {
            Some(cols[2].trim().parse().map_err(|_| {
                SnapshotError::IndexParse(format!("line {}: bad byte_offset", line_no + 2))
            })?)
        };
        let byte_length = if cols[3].trim().is_empty() {
            None
        } else {
            Some(cols[3].trim().parse().map_err(|_| {
                SnapshotError::IndexParse(format!("line {}: bad byte_length", line_no + 2))
            })?)
        };
        let decode_hint = if cols[4].trim().is_empty() {
            None
        } else {
            Some(cols[4].trim().to_string())
        };

        let record = ManifestRecord {
            sample_id,
            location,
            byte_offset,
            byte_length,
            decode_hint,
        };
        record
            .validate()
            .map_err(|e| SnapshotError::IndexParse(format!("line {}: {e}", line_no + 2)))?;
        records.push(record);
    }

    records.sort_by_key(|r| r.sample_id);
    ensure_sequential_sample_ids(&records)?;

    let canonical = canonicalize_manifest_bytes(&records);
    Ok((records, canonical))
}

fn snapshot_env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(raw) => matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    }
}

fn is_video_extension(path: &std::path::Path) -> bool {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase());
    matches!(
        ext.as_deref(),
        Some("mp4")
            | Some("mov")
            | Some("avi")
            | Some("mkv")
            | Some("webm")
            | Some("m4v")
            | Some("mpg")
            | Some("mpeg")
    )
}

fn codec_from_extension(path: &std::path::Path) -> &'static str {
    match path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("webm") => "vp9",
        Some("avi") | Some("mpg") | Some("mpeg") => "mpeg4",
        _ => "h264",
    }
}

fn is_supported_video_codec(codec: &str) -> bool {
    matches!(
        codec.to_ascii_lowercase().as_str(),
        "h264" | "hevc" | "vp9" | "mpeg4" | "av1" | "h263" | "mjpeg"
    )
}

fn probe_video_with_ffprobe(
    path: &std::path::Path,
) -> Result<Option<(u64, String)>, SnapshotError> {
    let out = std::process::Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("v:0")
        .arg("-show_entries")
        .arg("stream=codec_name,nb_frames")
        .arg("-of")
        .arg("csv=p=0")
        .arg(path.as_os_str())
        .output();

    let out = match out {
        Ok(v) => v,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(SnapshotError::IndexIo(format!(
                "ffprobe invocation failed for {}: {e}",
                path.display()
            )))
        }
    };

    if !out.status.success() {
        return Ok(Some((1, "corrupt".to_string())));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().next().map(str::trim).unwrap_or_default();
    if line.is_empty() {
        return Ok(Some((1, "corrupt".to_string())));
    }
    let mut parts = line.split(',');
    let codec = parts.next().unwrap_or("").trim().to_ascii_lowercase();
    let frames_raw = parts.next().unwrap_or("").trim();
    let frames = frames_raw
        .parse::<u64>()
        .ok()
        .filter(|v| *v > 0)
        .unwrap_or(1);
    if codec.is_empty() {
        return Ok(Some((frames, "corrupt".to_string())));
    }
    Ok(Some((frames, codec)))
}

fn stage1_video_decode_hint_for_local(
    path: &std::path::Path,
) -> Result<Option<String>, SnapshotError> {
    if !snapshot_env_bool("MX8_VIDEO_STAGE1_INDEX", false) {
        return Ok(None);
    }
    if !is_video_extension(path) {
        return Ok(None);
    }
    let disable_ffprobe = snapshot_env_bool("MX8_VIDEO_STAGE1_DISABLE_FFPROBE", false);
    let probed = if disable_ffprobe {
        None
    } else {
        probe_video_with_ffprobe(path)?
    };

    let (frames, codec) = match probed {
        Some((frames, codec)) if codec == "corrupt" => {
            return Ok(Some(
                "mx8:video;frames=1;stream_id=0;corrupt=true".to_string(),
            ))
        }
        Some((frames, codec)) => (frames, codec),
        None => {
            let bytes_per_frame_estimate =
                std::env::var("MX8_VIDEO_STAGE1_BYTES_PER_FRAME_ESTIMATE")
                    .ok()
                    .and_then(|v| v.trim().parse::<u64>().ok())
                    .filter(|v| *v > 0)
                    .unwrap_or(50_000);
            let file_len = std::fs::metadata(path)
                .map_err(|e| {
                    SnapshotError::IndexIo(format!("stat {} failed: {e}", path.display()))
                })?
                .len();
            let est_frames = file_len
                .max(1)
                .saturating_div(bytes_per_frame_estimate)
                .max(1);
            (est_frames, codec_from_extension(path).to_string())
        }
    };

    let codec_field = if is_supported_video_codec(&codec) {
        codec
    } else {
        "unsupported".to_string()
    };
    Ok(Some(format!(
        "mx8:video;frames={frames};stream_id=0;codec={codec_field}"
    )))
}

#[cfg(feature = "s3")]
fn stage1_video_decode_hint_for_s3_key(
    key: &str,
    object_size_bytes: Option<i64>,
) -> Option<String> {
    if !snapshot_env_bool("MX8_VIDEO_STAGE1_INDEX", false) {
        return None;
    }
    let path = std::path::Path::new(key);
    if !is_video_extension(path) {
        return None;
    }
    let bytes_per_frame_estimate = std::env::var("MX8_VIDEO_STAGE1_BYTES_PER_FRAME_ESTIMATE")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(50_000);
    let size_u64 = object_size_bytes
        .and_then(|v| u64::try_from(v).ok())
        .unwrap_or(1);
    let est_frames = size_u64
        .max(1)
        .saturating_div(bytes_per_frame_estimate)
        .max(1);
    let codec = codec_from_extension(path);
    let codec_field = if is_supported_video_codec(codec) {
        codec.to_string()
    } else {
        "unsupported".to_string()
    };
    Some(format!(
        "mx8:video;frames={est_frames};stream_id=0;codec={codec_field}"
    ))
}

fn index_local_prefix(
    base: &str,
    recursive: bool,
) -> Result<(Vec<ManifestRecord>, Vec<u8>), SnapshotError> {
    let scan_started = Instant::now();
    let root = std::path::PathBuf::from(base.trim());
    let manifest_path = root.join("_mx8").join("manifest.tsv");
    if manifest_path.exists() {
        let bytes = std::fs::read(&manifest_path).map_err(|e| {
            SnapshotError::IndexIo(format!(
                "read precomputed manifest failed: {}: {e}",
                manifest_path.display()
            ))
        })?;
        info!(
            target: "mx8_proof",
            event = "snapshot_index_precomputed_manifest",
            base = %base,
            manifest_path = %manifest_path.display(),
            "indexed from precomputed manifest"
        );
        return parse_canonical_manifest_tsv_bytes(&bytes);
    }

    // Fallback: index directory by walking files (no byte ranges, no labels).
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    let mut dirs_scanned: u64 = 0;
    let mut stack: Vec<(std::path::PathBuf, usize)> = vec![(root.clone(), 0)];
    while let Some((dir, depth)) = stack.pop() {
        dirs_scanned = dirs_scanned.saturating_add(1);
        for entry in std::fs::read_dir(&dir).map_err(|e| {
            SnapshotError::IndexIo(format!("read_dir failed: {}: {e}", dir.display()))
        })? {
            let entry = entry.map_err(|e| SnapshotError::IndexIo(format!("{e}")))?;
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == "_mx8" || name == "shards" {
                continue;
            }
            let meta = entry
                .metadata()
                .map_err(|e| SnapshotError::IndexIo(format!("{e}")))?;
            if meta.is_dir() {
                if recursive {
                    stack.push((path, depth.saturating_add(1)));
                }
            } else if meta.is_file() {
                files.push(path);
            }
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(SnapshotError::IndexParse(
            "local index produced zero files".to_string(),
        ));
    }

    let video_stage1_enabled = snapshot_env_bool("MX8_VIDEO_STAGE1_INDEX", false);
    let mut video_candidates_total: u64 = 0;
    let mut video_hints_emitted_total: u64 = 0;
    let mut video_codec_unsupported_total: u64 = 0;
    let mut video_probe_corrupt_total: u64 = 0;

    let mut records: Vec<ManifestRecord> = Vec::with_capacity(files.len());
    for (i, path) in files.into_iter().enumerate() {
        let abs = path.canonicalize().unwrap_or(path);
        if video_stage1_enabled && is_video_extension(&abs) {
            video_candidates_total = video_candidates_total.saturating_add(1);
        }
        let decode_hint = stage1_video_decode_hint_for_local(&abs)?;
        if let Some(hint) = decode_hint.as_deref() {
            if hint.starts_with("mx8:video;") {
                video_hints_emitted_total = video_hints_emitted_total.saturating_add(1);
            }
            if hint.contains("codec=unsupported") {
                video_codec_unsupported_total = video_codec_unsupported_total.saturating_add(1);
            }
            if hint.contains("corrupt=true") {
                video_probe_corrupt_total = video_probe_corrupt_total.saturating_add(1);
            }
        }
        let record = ManifestRecord {
            sample_id: i as u64,
            location: abs.display().to_string(),
            byte_offset: None,
            byte_length: None,
            decode_hint,
        };
        record
            .validate()
            .map_err(|e| SnapshotError::IndexParse(format!("indexed record invalid: {e}")))?;
        records.push(record);
    }

    let canonical = canonicalize_manifest_bytes(&records);
    info!(
        target: "mx8_proof",
        event = "snapshot_index_summary",
        backend = "local",
        base = %base,
        recursive = recursive,
        objects_indexed = records.len() as u64,
        dirs_scanned = dirs_scanned,
        scan_ms = scan_started.elapsed().as_millis() as u64,
        "snapshot index summary"
    );
    info!(
        target: "mx8_proof",
        event = "snapshot_video_stage1_metadata_summary",
        backend = "local",
        base = %base,
        recursive = recursive,
        video_stage1_enabled = video_stage1_enabled,
        video_candidates_total = video_candidates_total,
        video_hints_emitted_total = video_hints_emitted_total,
        video_codec_unsupported_total = video_codec_unsupported_total,
        video_probe_corrupt_total = video_probe_corrupt_total,
        "video stage1 metadata extraction summary"
    );
    Ok((records, canonical))
}

#[cfg(feature = "s3")]
fn index_s3_prefix(
    base: &str,
    recursive: bool,
    materialize_manifest_bytes: bool,
) -> Result<IndexedManifest, SnapshotError> {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum S3LabelMode {
        None,
        ImageFolder,
        Auto,
    }

    fn parse_label_mode() -> Result<S3LabelMode, SnapshotError> {
        match std::env::var("MX8_S3_LABEL_MODE") {
            Ok(v) => {
                let s = v.trim().to_ascii_lowercase();
                match s.as_str() {
                    "none" | "off" | "false" | "0" => Ok(S3LabelMode::None),
                    "imagefolder" | "image_folder" | "image-folder" => Ok(S3LabelMode::ImageFolder),
                    "auto" | "" => Ok(S3LabelMode::Auto),
                    _ => Err(SnapshotError::S3Index(format!(
                        "invalid MX8_S3_LABEL_MODE={v:?} (expected: auto|none|imagefolder)"
                    ))),
                }
            }
            Err(std::env::VarError::NotPresent) => Ok(S3LabelMode::Auto),
            Err(e) => Err(SnapshotError::S3Index(format!(
                "read env var MX8_S3_LABEL_MODE failed: {e}"
            ))),
        }
    }

    fn imagefolder_label_for_key(prefix: &str, key: &str) -> Option<String> {
        // If base is s3://bucket/prefix[/], we interpret "ImageFolder" labels as:
        //   prefix/<label>/<file...>
        //
        // i.e. the first path segment after prefix.
        let p = prefix.trim_matches('/');
        let mut rest = key.trim_start_matches('/');
        if !p.is_empty() {
            // `list_objects_v2(prefix=p)` typically yields keys that start with `p`,
            // but be defensive.
            if let Some(after) = rest.strip_prefix(p) {
                rest = after.strip_prefix('/').unwrap_or(after);
            }
        }
        let rest = rest.trim_start_matches('/');
        let (label, _tail) = rest.split_once('/')?;
        let label = label.trim();
        if label.is_empty() {
            return None;
        }
        Some(label.to_string())
    }

    fn parse_canonical_manifest_tsv_bytes(
        bytes: &[u8],
    ) -> Result<(Vec<ManifestRecord>, Vec<u8>), SnapshotError> {
        let s = std::str::from_utf8(bytes)
            .map_err(|e| SnapshotError::S3Index(format!("manifest not utf-8: {e}")))?;

        let mut lines = s.lines();
        let first = lines
            .by_ref()
            .find(|l| !l.trim().is_empty())
            .ok_or_else(|| SnapshotError::S3Index("empty manifest".to_string()))?;

        let Some((k, v)) = first.split_once('=') else {
            return Err(SnapshotError::S3Index(
                "manifest header missing schema_version".to_string(),
            ));
        };
        if k.trim() != "schema_version" {
            return Err(SnapshotError::S3Index(
                "manifest header must be schema_version=<n>".to_string(),
            ));
        }
        let schema_version: u32 = v
            .trim()
            .parse()
            .map_err(|_| SnapshotError::S3Index("invalid schema_version".to_string()))?;
        if schema_version != MANIFEST_SCHEMA_VERSION {
            return Err(SnapshotError::S3Index(format!(
                "unsupported schema_version {schema_version}"
            )));
        }

        let mut records: Vec<ManifestRecord> = Vec::new();
        for (i, raw) in lines.enumerate() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() < 2 {
                return Err(SnapshotError::S3Index(format!(
                    "line {}: expected at least 2 columns",
                    i + 2
                )));
            }

            let sample_id: u64 = cols[0]
                .trim()
                .parse()
                .map_err(|_| SnapshotError::S3Index(format!("line {}: bad sample_id", i + 2)))?;
            let location = cols[1].trim().to_string();
            if location.contains('\n') || location.contains('\t') {
                return Err(SnapshotError::S3Index(format!(
                    "line {}: location must not contain tabs/newlines",
                    i + 2
                )));
            }

            let mut byte_offset: Option<u64> = None;
            let mut byte_length: Option<u64> = None;
            let mut decode_hint: Option<String> = None;

            if cols.len() >= 4 {
                if !cols[2].trim().is_empty() || !cols[3].trim().is_empty() {
                    byte_offset = Some(cols[2].trim().parse().map_err(|_| {
                        SnapshotError::S3Index(format!("line {}: bad byte_offset", i + 2))
                    })?);
                    byte_length = Some(cols[3].trim().parse().map_err(|_| {
                        SnapshotError::S3Index(format!("line {}: bad byte_length", i + 2))
                    })?);
                }
            } else if cols.len() == 3 {
                return Err(SnapshotError::S3Index(format!(
                    "line {}: byte_offset and byte_length must be set together",
                    i + 2
                )));
            }

            if cols.len() >= 5 && !cols[4].trim().is_empty() {
                let hint = cols[4].trim().to_string();
                if hint.contains('\n') || hint.contains('\t') {
                    return Err(SnapshotError::S3Index(format!(
                        "line {}: decode_hint must not contain tabs/newlines",
                        i + 2
                    )));
                }
                decode_hint = Some(hint);
            }

            let record = ManifestRecord {
                sample_id,
                location,
                byte_offset,
                byte_length,
                decode_hint,
            };
            record.validate().map_err(|e| {
                SnapshotError::S3Index(format!(
                    "line {}: indexed record failed validation: {e}",
                    i + 2
                ))
            })?;
            records.push(record);
        }

        records.sort_by_key(|r| r.sample_id);
        ensure_sequential_sample_ids(&records)
            .map_err(|e| SnapshotError::S3Index(format!("manifest invariant violated: {e}")))?;

        let canonical = canonicalize_manifest_bytes(&records);
        Ok((records, canonical))
    }

    fn parse_env_bool(key: &str) -> Result<Option<bool>, SnapshotError> {
        match std::env::var(key) {
            Ok(v) => {
                let s = v.trim().to_ascii_lowercase();
                let b = match s.as_str() {
                    "1" | "true" | "yes" | "y" | "on" => true,
                    "0" | "false" | "no" | "n" | "off" => false,
                    _ => {
                        return Err(SnapshotError::S3Index(format!(
                            "invalid boolean env var {}={:?} (expected true/false/1/0)",
                            key, v
                        )))
                    }
                };
                Ok(Some(b))
            }
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(e) => Err(SnapshotError::S3Index(format!(
                "read env var {key} failed: {e}"
            ))),
        }
    }

    fn parse_env_usize(key: &str, default: usize) -> Result<usize, SnapshotError> {
        match std::env::var(key) {
            Ok(v) => {
                let n = v.trim().parse::<usize>().map_err(|_| {
                    SnapshotError::S3Index(format!(
                        "invalid usize env var {}={:?} (expected positive integer)",
                        key, v
                    ))
                })?;
                if n == 0 {
                    return Err(SnapshotError::S3Index(format!(
                        "invalid {}=0 (must be > 0)",
                        key
                    )));
                }
                Ok(n)
            }
            Err(std::env::VarError::NotPresent) => Ok(default),
            Err(e) => Err(SnapshotError::S3Index(format!(
                "read env var {key} failed: {e}"
            ))),
        }
    }

    struct SpillDir {
        path: std::path::PathBuf,
    }

    impl Drop for SpillDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn new_spill_dir() -> Result<SpillDir, SnapshotError> {
        let mut path = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        path.push(format!("mx8-snapshot-spill-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&path).map_err(|e| {
            SnapshotError::S3Index(format!(
                "failed to create spill directory {}: {e}",
                path.display()
            ))
        })?;
        Ok(SpillDir { path })
    }

    fn write_key_len_prefixed(
        writer: &mut std::io::BufWriter<std::fs::File>,
        key: &str,
    ) -> Result<(), SnapshotError> {
        let bytes = key.as_bytes();
        let len_u32 = u32::try_from(bytes.len())
            .map_err(|_| SnapshotError::S3Index("s3 object key too long".to_string()))?;
        use std::io::Write as _;
        writer
            .write_all(&len_u32.to_le_bytes())
            .map_err(|e| SnapshotError::S3Index(format!("spill write length failed: {e}")))?;
        writer
            .write_all(bytes)
            .map_err(|e| SnapshotError::S3Index(format!("spill write payload failed: {e}")))?;
        Ok(())
    }

    fn read_key_len_prefixed(
        reader: &mut std::io::BufReader<std::fs::File>,
    ) -> Result<Option<String>, SnapshotError> {
        use std::io::Read as _;
        let mut len = [0u8; 4];
        let n = reader
            .read(&mut len)
            .map_err(|e| SnapshotError::S3Index(format!("spill read length failed: {e}")))?;
        if n == 0 {
            return Ok(None);
        }
        if n != len.len() {
            return Err(SnapshotError::S3Index(
                "truncated spill file while reading key length".to_string(),
            ));
        }
        let key_len = u32::from_le_bytes(len) as usize;
        let mut key = vec![0u8; key_len];
        reader
            .read_exact(&mut key)
            .map_err(|e| SnapshotError::S3Index(format!("spill read key bytes failed: {e}")))?;
        let key = String::from_utf8(key)
            .map_err(|e| SnapshotError::S3Index(format!("spill key is not utf-8: {e}")))?;
        Ok(Some(key))
    }

    fn spill_sorted_run(
        spill_dir: &SpillDir,
        run_idx: usize,
        keys: &mut Vec<String>,
    ) -> Result<std::path::PathBuf, SnapshotError> {
        keys.sort();
        let path = spill_dir.path.join(format!("run-{run_idx:06}.bin"));
        let file = std::fs::File::create(&path)
            .map_err(|e| SnapshotError::S3Index(format!("create spill run failed: {e}")))?;
        let mut writer = std::io::BufWriter::new(file);
        for key in keys.iter() {
            write_key_len_prefixed(&mut writer, key)?;
        }
        use std::io::Write as _;
        writer
            .flush()
            .map_err(|e| SnapshotError::S3Index(format!("flush spill run failed: {e}")))?;
        keys.clear();
        Ok(path)
    }

    fn parse_s3_base(url: &str) -> Result<(String, String), SnapshotError> {
        // url: s3://bucket/prefix[/]
        let rest = url
            .strip_prefix("s3://")
            .ok_or_else(|| SnapshotError::S3Index(format!("invalid s3 url: {url}")))?;
        let s = rest.trim().trim_matches('/');
        let mut it = s.splitn(2, '/');
        let bucket = it.next().unwrap_or("").trim();
        if bucket.is_empty() {
            return Err(SnapshotError::S3Index(format!(
                "invalid s3 url (missing bucket): {url}"
            )));
        }
        let prefix = it.next().unwrap_or("").trim_matches('/').to_string();
        Ok((bucket.to_string(), prefix))
    }

    fn block_on<Fut>(fut: Fut) -> Result<Fut::Output, SnapshotError>
    where
        Fut: std::future::Future,
    {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => Ok(tokio::task::block_in_place(|| handle.block_on(fut))),
            Err(_) => {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| {
                        SnapshotError::S3Index(format!("tokio runtime init failed: {e}"))
                    })?;
                Ok(rt.block_on(fut))
            }
        }
    }

    async fn client_from_env() -> Result<aws_sdk_s3::Client, SnapshotError> {
        let cfg = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;

        let endpoint_url: Option<String> = std::env::var("MX8_S3_ENDPOINT_URL").ok();
        let force_path_style = match parse_env_bool("MX8_S3_FORCE_PATH_STYLE")? {
            Some(v) => v,
            None => endpoint_url.is_some(),
        };

        let mut b = aws_sdk_s3::config::Builder::from(&cfg);
        if let Some(url) = endpoint_url {
            b = b.endpoint_url(url);
        }
        if force_path_style {
            b = b.force_path_style(true);
        }

        Ok(aws_sdk_s3::Client::from_conf(b.build()))
    }

    let (bucket, prefix) = parse_s3_base(base)?;
    let list_prefix = if prefix.is_empty() {
        None
    } else {
        Some(format!("{}/", prefix.trim_end_matches('/')))
    };
    let client = block_on(client_from_env())??;
    let label_mode = parse_label_mode()?;
    let video_stage1_enabled = snapshot_env_bool("MX8_VIDEO_STAGE1_INDEX", false);

    // Fast path: if the dataset prefix is already packed, use the precomputed canonical manifest.
    //
    // Convention: `s3://bucket/prefix/_mx8/manifest.tsv` (or `_mx8/manifest.tsv` at bucket root if prefix is empty).
    let precomputed_key = if prefix.is_empty() {
        "_mx8/manifest.tsv".to_string()
    } else {
        format!("{prefix}/_mx8/manifest.tsv")
    };
    let precomputed = block_on(async {
        let head = client
            .head_object()
            .bucket(&bucket)
            .key(&precomputed_key)
            .send()
            .await;
        match head {
            Ok(_) => {
                let obj = client
                    .get_object()
                    .bucket(&bucket)
                    .key(&precomputed_key)
                    .send()
                    .await
                    .map_err(|e| {
                        SnapshotError::S3Index(format!(
                            "get_object precomputed manifest failed: {e:?}"
                        ))
                    })?;
                let collected = obj
                    .body
                    .collect()
                    .await
                    .map_err(|e| SnapshotError::S3Index(format!("collect failed: {e:?}")))?;
                Ok::<_, SnapshotError>(Some(collected.into_bytes().to_vec()))
            }
            Err(_) => Ok::<_, SnapshotError>(None),
        }
    })??;

    if let Some(bytes) = precomputed {
        let (records, canonical) = parse_canonical_manifest_tsv_bytes(&bytes)?;
        info!(
            target: "mx8_proof",
            event = "snapshot_index_precomputed_manifest",
            base = %base,
            bucket = %bucket,
            prefix = %prefix,
            key = %precomputed_key,
            objects = records.len() as u64,
            "indexed from precomputed manifest"
        );
        return indexed_manifest_from_bytes(records, canonical, materialize_manifest_bytes);
    }

    let scan_started = Instant::now();
    let use_external_sort = parse_env_bool("MX8_SNAPSHOT_S3_EXTERNAL_SORT")?.unwrap_or(false);

    let mut sink = CanonicalManifestSink::new(materialize_manifest_bytes)?;
    let mut sample_id: u64 = 0;
    let mut pages_scanned: u64 = 0;

    if use_external_sort {
        let spill_keys_per_run = parse_env_usize("MX8_SNAPSHOT_S3_SPILL_KEYS_PER_RUN", 100_000)?;
        let spill_dir = new_spill_dir()?;
        let mut run_files: Vec<std::path::PathBuf> = Vec::new();
        let mut run_keys: Vec<String> = Vec::with_capacity(spill_keys_per_run);
        let mut run_idx: usize = 0;

        let mut label_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut saw_all_imagefolder = true;
        let mut objects_seen: u64 = 0;
        let mut video_candidates_total: u64 = 0;
        let mut video_hints_emitted_total: u64 = 0;
        let mut video_codec_unsupported_total: u64 = 0;

        block_on(async {
            let mut token: Option<String> = None;
            loop {
                let mut req = client.list_objects_v2().bucket(&bucket);
                if let Some(p) = list_prefix.as_deref() {
                    req = req.prefix(p);
                }
                if !recursive {
                    req = req.delimiter("/");
                }
                if let Some(t) = token.as_deref() {
                    req = req.continuation_token(t);
                }
                let resp = req.send().await.map_err(|e| {
                    SnapshotError::S3Index(format!("list_objects_v2 failed while indexing: {e:?}"))
                })?;
                pages_scanned = pages_scanned.saturating_add(1);

                if let Some(contents) = resp.contents {
                    for obj in contents {
                        let Some(k) = obj.key else { continue };
                        if k.ends_with('/') {
                            continue;
                        }
                        objects_seen = objects_seen.saturating_add(1);
                        if video_stage1_enabled && is_video_extension(std::path::Path::new(&k)) {
                            video_candidates_total = video_candidates_total.saturating_add(1);
                        }
                        if label_mode != S3LabelMode::None {
                            let label = imagefolder_label_for_key(&prefix, &k);
                            if let Some(l) = label {
                                label_set.insert(l);
                            } else {
                                saw_all_imagefolder = false;
                            }
                        }
                        run_keys.push(k);
                        if run_keys.len() >= spill_keys_per_run {
                            let path = spill_sorted_run(&spill_dir, run_idx, &mut run_keys)?;
                            run_files.push(path);
                            run_idx = run_idx.saturating_add(1);
                        }
                    }
                }

                if resp.is_truncated.unwrap_or(false) {
                    token = resp.next_continuation_token;
                    if token.is_none() {
                        break;
                    }
                } else {
                    break;
                }
            }
            Ok::<_, SnapshotError>(())
        })??;

        if !run_keys.is_empty() {
            let path = spill_sorted_run(&spill_dir, run_idx, &mut run_keys)?;
            run_files.push(path);
        }

        let use_imagefolder_labels = match label_mode {
            S3LabelMode::None => false,
            S3LabelMode::ImageFolder => {
                if !saw_all_imagefolder {
                    return Err(SnapshotError::S3Index(
                        "MX8_S3_LABEL_MODE=imagefolder requires all keys to match prefix/<label>/<file>"
                            .to_string(),
                    ));
                }
                true
            }
            S3LabelMode::Auto => saw_all_imagefolder && label_set.len() >= 2,
        };

        let mut label_map: std::collections::HashMap<String, u64> =
            std::collections::HashMap::new();
        if use_imagefolder_labels {
            for (i, label) in label_set.into_iter().enumerate() {
                label_map.insert(label, i as u64);
            }
        }

        let mut readers: Vec<std::io::BufReader<std::fs::File>> =
            Vec::with_capacity(run_files.len());
        let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(String, usize)>> =
            std::collections::BinaryHeap::new();

        for (idx, path) in run_files.iter().enumerate() {
            let file = std::fs::File::open(path)
                .map_err(|e| SnapshotError::S3Index(format!("open spill run failed: {e}")))?;
            let mut reader = std::io::BufReader::new(file);
            if let Some(key) = read_key_len_prefixed(&mut reader)? {
                heap.push(std::cmp::Reverse((key, idx)));
            }
            readers.push(reader);
        }

        let mut prev_key: Option<String> = None;
        while let Some(std::cmp::Reverse((key, run_idx))) = heap.pop() {
            if let Some(prev) = prev_key.as_deref() {
                if key.as_str() <= prev {
                    return Err(SnapshotError::S3Index(format!(
                        "merged key stream is not strictly increasing ({prev:?} then {key:?})"
                    )));
                }
            }
            prev_key = Some(key.clone());

            let location = format!("s3://{bucket}/{key}");
            if location.contains('\n') || location.contains('\t') {
                return Err(SnapshotError::S3Index(format!(
                    "s3 object url contains tabs/newlines (sample_id={}): {:?}",
                    sample_id, location
                )));
            }
            let decode_hint =
                if let Some(video_hint) = stage1_video_decode_hint_for_s3_key(&key, None) {
                    Some(video_hint)
                } else if use_imagefolder_labels {
                    let label = imagefolder_label_for_key(&prefix, &key).ok_or_else(|| {
                        SnapshotError::S3Index(format!(
                        "label missing under imagefolder mode (sample_id={sample_id}, key={key})"
                    ))
                    })?;
                    let label_id = *label_map.get(&label).ok_or_else(|| {
                        SnapshotError::S3Index(format!(
                        "label not present in label map (sample_id={sample_id}, label={label:?})"
                    ))
                    })?;
                    Some(format!("mx8:vision:imagefolder;label_id={label_id}"))
                } else {
                    None
                };
            if let Some(hint) = decode_hint.as_deref() {
                if hint.starts_with("mx8:video;") {
                    video_hints_emitted_total = video_hints_emitted_total.saturating_add(1);
                }
                if hint.contains("codec=unsupported") {
                    video_codec_unsupported_total = video_codec_unsupported_total.saturating_add(1);
                }
            }
            let record = ManifestRecord {
                sample_id,
                location,
                byte_offset: None,
                byte_length: None,
                decode_hint,
            };
            record.validate().map_err(|e| {
                SnapshotError::S3Index(format!("indexed record failed validation: {e}"))
            })?;
            sink.append_record(&record)?;
            sample_id = sample_id.saturating_add(1);

            let reader = readers.get_mut(run_idx).ok_or_else(|| {
                SnapshotError::S3Index(format!(
                    "internal spill reader index out of bounds: {run_idx}"
                ))
            })?;
            if let Some(next_key) = read_key_len_prefixed(reader)? {
                heap.push(std::cmp::Reverse((next_key, run_idx)));
            }
        }

        if sample_id != objects_seen {
            return Err(SnapshotError::S3Index(format!(
                "spill/merge object count mismatch (seen={objects_seen}, emitted={sample_id})"
            )));
        }

        info!(
            target: "mx8_proof",
            event = "snapshot_index_spill_merge",
            bucket = %bucket,
            prefix = %prefix,
            recursive = recursive,
            spill_runs = run_files.len() as u64,
            spill_keys_per_run = spill_keys_per_run as u64,
            "used external spill/merge while indexing s3 prefix"
        );

        let label_mode_name = if use_imagefolder_labels {
            "imagefolder"
        } else {
            "none"
        };
        if sample_id == 0 {
            warn!(
                target: "mx8_proof",
                event = "snapshot_index_empty",
                base = %base,
                bucket = %bucket,
                prefix = %prefix,
                recursive = recursive,
                "s3 index produced zero objects"
            );
        }
        let scan_ms = scan_started.elapsed().as_millis() as u64;
        info!(
            target: "mx8_proof",
            event = "snapshot_indexed",
            base = %base,
            bucket = %bucket,
            prefix = %prefix,
            recursive = recursive,
            label_mode = label_mode_name,
            objects = sample_id,
            "indexed s3 prefix"
        );
        info!(
            target: "mx8_proof",
            event = "snapshot_index_summary",
            backend = "s3",
            bucket = %bucket,
            prefix = %prefix,
            recursive = recursive,
            label_mode = label_mode_name,
            external_sort = use_external_sort,
            pages_scanned = pages_scanned,
            objects_indexed = sample_id,
            scan_ms = scan_ms,
            "snapshot index summary"
        );
        info!(
            target: "mx8_proof",
            event = "snapshot_video_stage1_metadata_summary",
            backend = "s3",
            bucket = %bucket,
            prefix = %prefix,
            recursive = recursive,
            external_sort = true,
            video_stage1_enabled = video_stage1_enabled,
            video_candidates_total = video_candidates_total,
            video_hints_emitted_total = video_hints_emitted_total,
            video_codec_unsupported_total = video_codec_unsupported_total,
            "video stage1 metadata extraction summary"
        );
        let output = sink.finish()?;
        return Ok(IndexedManifest {
            manifest_hash: output.manifest_hash,
            manifest_bytes: output.manifest_bytes,
            canonical_bytes: output.canonical_bytes,
            canonical_path: output.canonical_path,
        });
    }

    let mut label_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut saw_all_imagefolder = true;
    let mut video_candidates_total: u64 = 0;
    let mut video_hints_emitted_total: u64 = 0;
    let mut video_codec_unsupported_total: u64 = 0;
    if label_mode != S3LabelMode::None {
        block_on(async {
            let mut token: Option<String> = None;
            let mut prev_key: Option<String> = None;
            loop {
                let mut req = client.list_objects_v2().bucket(&bucket);
                if let Some(p) = list_prefix.as_deref() {
                    req = req.prefix(p);
                }
                if !recursive {
                    req = req.delimiter("/");
                }
                if let Some(t) = token.as_deref() {
                    req = req.continuation_token(t);
                }
                let resp = req.send().await.map_err(|e| {
                    SnapshotError::S3Index(format!(
                        "list_objects_v2 failed during label scan: {e:?}"
                    ))
                })?;
                pages_scanned = pages_scanned.saturating_add(1);

                if let Some(contents) = resp.contents {
                    for obj in contents {
                        let Some(k) = obj.key else { continue };
                        if k.ends_with('/') {
                            continue;
                        }
                        if let Some(prev) = prev_key.as_deref() {
                            if k.as_str() <= prev {
                                return Err(SnapshotError::S3Index(format!(
                                    "s3 list order is not strictly increasing ({prev:?} then {k:?}); aborting snapshot index"
                                )));
                            }
                        }
                        prev_key = Some(k.clone());
                        let label = imagefolder_label_for_key(&prefix, &k);
                        if let Some(l) = label {
                            label_set.insert(l);
                        } else {
                            saw_all_imagefolder = false;
                        }
                    }
                }

                if resp.is_truncated.unwrap_or(false) {
                    token = resp.next_continuation_token;
                    if token.is_none() {
                        break;
                    }
                } else {
                    break;
                }
            }
            Ok::<_, SnapshotError>(())
        })??;
    }

    let use_imagefolder_labels = match label_mode {
        S3LabelMode::None => false,
        S3LabelMode::ImageFolder => {
            if !saw_all_imagefolder {
                return Err(SnapshotError::S3Index(
                    "MX8_S3_LABEL_MODE=imagefolder requires all keys to match prefix/<label>/<file>"
                        .to_string(),
                ));
            }
            true
        }
        S3LabelMode::Auto => saw_all_imagefolder && label_set.len() >= 2,
    };

    let mut label_map: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    if use_imagefolder_labels {
        for (i, label) in label_set.into_iter().enumerate() {
            label_map.insert(label, i as u64);
        }
    }

    block_on(async {
        let mut token: Option<String> = None;
        let mut prev_key: Option<String> = None;
        loop {
            let mut req = client.list_objects_v2().bucket(&bucket);
            if let Some(p) = list_prefix.as_deref() {
                req = req.prefix(p);
            }
            if !recursive {
                req = req.delimiter("/");
            }
            if let Some(t) = token.as_deref() {
                req = req.continuation_token(t);
            }
            let resp = req.send().await.map_err(|e| {
                SnapshotError::S3Index(format!("list_objects_v2 failed while indexing: {e:?}"))
            })?;
            pages_scanned = pages_scanned.saturating_add(1);

            if let Some(contents) = resp.contents {
                for obj in contents {
                    let key_size = obj.size;
                    let Some(k) = obj.key else { continue };
                    if k.ends_with('/') {
                        continue;
                    }
                    if video_stage1_enabled && is_video_extension(std::path::Path::new(&k)) {
                        video_candidates_total = video_candidates_total.saturating_add(1);
                    }
                    if let Some(prev) = prev_key.as_deref() {
                        if k.as_str() <= prev {
                            return Err(SnapshotError::S3Index(format!(
                                "s3 list order is not strictly increasing ({prev:?} then {k:?}); aborting snapshot index"
                            )));
                        }
                    }
                    prev_key = Some(k.clone());
                    let location = format!("s3://{bucket}/{k}");
                    if location.contains('\n') || location.contains('\t') {
                        return Err(SnapshotError::S3Index(format!(
                            "s3 object url contains tabs/newlines (sample_id={}): {:?}",
                            sample_id, location
                        )));
                    }
                    let decode_hint = if let Some(video_hint) =
                        stage1_video_decode_hint_for_s3_key(&k, key_size)
                    {
                        Some(video_hint)
                    } else if use_imagefolder_labels {
                        let label = imagefolder_label_for_key(&prefix, &k).ok_or_else(|| {
                                SnapshotError::S3Index(format!(
                                    "label missing under imagefolder mode (sample_id={sample_id}, key={k})"
                                ))
                            })?;
                        let label_id = *label_map.get(&label).ok_or_else(|| {
                                SnapshotError::S3Index(format!(
                                    "label not present in label map (sample_id={sample_id}, label={label:?})"
                                ))
                            })?;
                        Some(format!("mx8:vision:imagefolder;label_id={label_id}"))
                    } else {
                        None
                    };
                    if let Some(hint) = decode_hint.as_deref() {
                        if hint.starts_with("mx8:video;") {
                            video_hints_emitted_total = video_hints_emitted_total.saturating_add(1);
                        }
                        if hint.contains("codec=unsupported") {
                            video_codec_unsupported_total =
                                video_codec_unsupported_total.saturating_add(1);
                        }
                    }

                    let record = ManifestRecord {
                        sample_id,
                        location,
                        byte_offset: None,
                        byte_length: None,
                        decode_hint,
                    };
                    record.validate().map_err(|e| {
                        SnapshotError::S3Index(format!("indexed record failed validation: {e}"))
                    })?;
                    sink.append_record(&record)?;
                    sample_id = sample_id.saturating_add(1);
                }
            }

            if resp.is_truncated.unwrap_or(false) {
                token = resp.next_continuation_token;
                if token.is_none() {
                    break;
                }
            } else {
                break;
            }
        }
        Ok::<_, SnapshotError>(())
    })??;

    if sample_id == 0 {
        warn!(
            target: "mx8_proof",
            event = "snapshot_index_empty",
            base = %base,
            bucket = %bucket,
            prefix = %prefix,
            recursive = recursive,
            "s3 index produced zero objects"
        );
    }

    let label_mode_name = if use_imagefolder_labels {
        "imagefolder"
    } else {
        "none"
    };
    let scan_ms = scan_started.elapsed().as_millis() as u64;
    info!(
        target: "mx8_proof",
        event = "snapshot_indexed",
        base = %base,
        bucket = %bucket,
        prefix = %prefix,
        recursive = recursive,
        label_mode = label_mode_name,
        objects = sample_id,
        "indexed s3 prefix"
    );
    info!(
        target: "mx8_proof",
        event = "snapshot_index_summary",
        backend = "s3",
        bucket = %bucket,
        prefix = %prefix,
        recursive = recursive,
        label_mode = label_mode_name,
        external_sort = use_external_sort,
        pages_scanned = pages_scanned,
        objects_indexed = sample_id,
        scan_ms = scan_ms,
        "snapshot index summary"
    );
    info!(
        target: "mx8_proof",
        event = "snapshot_video_stage1_metadata_summary",
        backend = "s3",
        bucket = %bucket,
        prefix = %prefix,
        recursive = recursive,
        external_sort = false,
        video_stage1_enabled = video_stage1_enabled,
        video_candidates_total = video_candidates_total,
        video_hints_emitted_total = video_hints_emitted_total,
        video_codec_unsupported_total = video_codec_unsupported_total,
        "video stage1 metadata extraction summary"
    );
    let output = sink.finish()?;
    Ok(IndexedManifest {
        manifest_hash: output.manifest_hash,
        manifest_bytes: output.manifest_bytes,
        canonical_bytes: output.canonical_bytes,
        canonical_path: output.canonical_path,
    })
}

fn load_dev_manifest_tsv(path: &PathBuf) -> Result<(Vec<ManifestRecord>, Vec<u8>), SnapshotError> {
    let bytes = std::fs::read(path)?;
    let text = String::from_utf8_lossy(&bytes);

    let mut records: Vec<ManifestRecord> = Vec::new();
    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 2 {
            return Err(SnapshotError::DevManifestParse(format!(
                "line {}: expected at least 2 columns (sample_id<TAB>location)",
                line_no + 1
            )));
        }

        let sample_id: u64 = cols[0].trim().parse().map_err(|_| {
            SnapshotError::DevManifestParse(format!("line {}: bad sample_id", line_no + 1))
        })?;
        let location = cols[1].trim().to_string();
        if location.contains('\n') || location.contains('\t') {
            return Err(SnapshotError::DevManifestParse(format!(
                "line {}: location must not contain tabs/newlines",
                line_no + 1
            )));
        }

        let mut byte_offset: Option<u64> = None;
        let mut byte_length: Option<u64> = None;
        let mut decode_hint: Option<String> = None;

        if cols.len() >= 4 {
            if !cols[2].trim().is_empty() || !cols[3].trim().is_empty() {
                byte_offset = Some(cols[2].trim().parse().map_err(|_| {
                    SnapshotError::DevManifestParse(format!(
                        "line {}: bad byte_offset",
                        line_no + 1
                    ))
                })?);
                byte_length = Some(cols[3].trim().parse().map_err(|_| {
                    SnapshotError::DevManifestParse(format!(
                        "line {}: bad byte_length",
                        line_no + 1
                    ))
                })?);
            }
        } else if cols.len() == 3 {
            return Err(SnapshotError::DevManifestParse(format!(
                "line {}: byte_offset and byte_length must be set together",
                line_no + 1
            )));
        }

        if cols.len() >= 5 && !cols[4].trim().is_empty() {
            let hint = cols[4].trim().to_string();
            if hint.contains('\n') || hint.contains('\t') {
                return Err(SnapshotError::DevManifestParse(format!(
                    "line {}: decode_hint must not contain tabs/newlines",
                    line_no + 1
                )));
            }
            decode_hint = Some(hint);
        }

        let record = ManifestRecord {
            sample_id,
            location,
            byte_offset,
            byte_length,
            decode_hint,
        };
        record.validate().map_err(|e| {
            SnapshotError::DevManifestInvariant(format!("line {}: {e}", line_no + 1))
        })?;
        records.push(record);
    }

    records.sort_by_key(|r| r.sample_id);
    ensure_sequential_sample_ids(&records)?;

    let canonical = canonicalize_manifest_bytes(&records);
    Ok((records, canonical))
}

fn ensure_sequential_sample_ids(records: &[ManifestRecord]) -> Result<(), SnapshotError> {
    for (i, r) in records.iter().enumerate() {
        let expected = i as u64;
        if r.sample_id != expected {
            return Err(SnapshotError::DevManifestInvariant(format!(
                "expected sample_id {expected} but found {}",
                r.sample_id
            )));
        }
    }
    Ok(())
}

fn canonicalize_manifest_bytes(records: &[ManifestRecord]) -> Vec<u8> {
    let mut out = Vec::with_capacity(records.len() * 64);
    out.extend_from_slice(format!("schema_version={MANIFEST_SCHEMA_VERSION}\n").as_bytes());
    for r in records {
        append_manifest_record_tsv_line(&mut out, r);
    }
    out
}

fn append_manifest_record_tsv_line(out: &mut Vec<u8>, r: &ManifestRecord) {
    out.extend_from_slice(manifest_record_tsv_line(r).as_bytes());
}

fn manifest_record_tsv_line(r: &ManifestRecord) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}\n",
        r.sample_id,
        r.location,
        r.byte_offset.map(|v| v.to_string()).unwrap_or_default(),
        r.byte_length.map(|v| v.to_string()).unwrap_or_default(),
        r.decode_hint.clone().unwrap_or_default()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mx8_manifest_store::fs::FsManifestStore;

    #[test]
    fn canonical_hash_is_deterministic() -> anyhow::Result<()> {
        let root = {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "mx8-snapshot-test-{}-{}",
                std::process::id(),
                mx8_manifest_store::sha256_hex(b"seed")
            ));
            std::fs::create_dir_all(&p)?;
            p
        };

        let store: Arc<dyn ManifestStore> = Arc::new(FsManifestStore::new(root));
        let cfg = SnapshotResolverConfig {
            dev_manifest_path: None,
            ..Default::default()
        };
        let resolver = SnapshotResolver::new(store, cfg);

        let records = vec![
            ManifestRecord {
                sample_id: 0,
                location: "a".to_string(),
                byte_offset: None,
                byte_length: None,
                decode_hint: None,
            },
            ManifestRecord {
                sample_id: 1,
                location: "b".to_string(),
                byte_offset: Some(0),
                byte_length: Some(10),
                decode_hint: Some("x".to_string()),
            },
        ];
        let canonical = canonicalize_manifest_bytes(&records);
        let h1 = mx8_manifest_store::sha256_hex(&canonical);
        let h2 = mx8_manifest_store::sha256_hex(&canonical);
        assert_eq!(h1, h2);

        // Silence unused var; resolver existence is what we care about in this smoke test.
        let _ = resolver;
        Ok(())
    }

    #[test]
    fn plain_link_creates_snapshot_then_reuses_current() -> anyhow::Result<()> {
        let root = {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "mx8-snapshot-e2e-{}-{}",
                std::process::id(),
                mx8_manifest_store::sha256_hex(b"root")
            ));
            std::fs::create_dir_all(&p)?;
            p
        };

        let manifest_path = root.join("dev_manifest.tsv");
        std::fs::write(&manifest_path, "0\ta\n1\tb\t0\t10\tx\n")?;

        let store: Arc<dyn ManifestStore> = Arc::new(FsManifestStore::new(root.clone()));
        let cfg = SnapshotResolverConfig {
            dev_manifest_path: Some(manifest_path),
            ..Default::default()
        };
        let resolver = SnapshotResolver::new(store.clone(), cfg);

        let first = resolver.resolve(
            "s3://bucket/prefix/",
            LockOwner {
                node_id: Some("test".to_string()),
            },
        )?;
        assert!(first.manifest_bytes_materialized);
        assert!(!first.manifest_hash.0.is_empty());
        assert!(!first.manifest_bytes.is_empty());

        // New resolver without dev_manifest_path should reuse the current pointer.
        let resolver2 = SnapshotResolver::new(
            store,
            SnapshotResolverConfig {
                dev_manifest_path: None,
                ..Default::default()
            },
        );
        let second = resolver2.resolve(
            "s3://bucket/prefix/",
            LockOwner {
                node_id: Some("test2".to_string()),
            },
        )?;
        assert_eq!(first.manifest_hash.0, second.manifest_hash.0);
        assert_eq!(first.manifest_bytes, second.manifest_bytes);
        Ok(())
    }

    #[test]
    fn plain_link_can_skip_manifest_bytes_materialization() -> anyhow::Result<()> {
        let root = {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "mx8-snapshot-nobytes-{}-{}",
                std::process::id(),
                mx8_manifest_store::sha256_hex(b"root")
            ));
            std::fs::create_dir_all(&p)?;
            p
        };

        let manifest_path = root.join("dev_manifest.tsv");
        std::fs::write(&manifest_path, "0\ta\n")?;

        let store: Arc<dyn ManifestStore> = Arc::new(FsManifestStore::new(root.clone()));
        let resolver = SnapshotResolver::new(
            store.clone(),
            SnapshotResolverConfig {
                dev_manifest_path: Some(manifest_path),
                materialize_manifest_bytes: false,
                ..Default::default()
            },
        );

        let resolved = resolver.resolve(
            "s3://bucket/prefix/",
            LockOwner {
                node_id: Some("test".to_string()),
            },
        )?;
        assert!(!resolved.manifest_bytes_materialized);
        assert!(resolved.manifest_bytes.is_empty());

        let bytes = store.get_manifest_bytes(&resolved.manifest_hash)?;
        assert!(!bytes.is_empty());
        Ok(())
    }

    #[test]
    fn missing_dev_manifest_path_surfaces_as_dev_manifest_io() {
        let root =
            std::env::temp_dir().join(format!("mx8-snapshot-missing-{}", std::process::id()));
        let store: Arc<dyn ManifestStore> = Arc::new(FsManifestStore::new(root));
        let resolver = SnapshotResolver::new(
            store,
            SnapshotResolverConfig {
                dev_manifest_path: Some(PathBuf::from("/no/such/file.tsv")),
                ..Default::default()
            },
        );

        let err = resolver
            .resolve(
                "s3://bucket/prefix/@refresh",
                LockOwner {
                    node_id: Some("test".to_string()),
                },
            )
            .unwrap_err();

        match err {
            SnapshotError::DevManifestIo(_) => {}
            other => panic!("expected DevManifestIo, got {other:?}"),
        }
    }

    #[test]
    fn local_index_non_recursive_excludes_nested_files() -> anyhow::Result<()> {
        let root =
            std::env::temp_dir().join(format!("mx8-snapshot-local-nonrec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("nested"))?;
        std::fs::write(root.join("a.txt"), b"a")?;
        std::fs::write(root.join("nested").join("b.txt"), b"b")?;

        let (records, _canonical) = index_local_prefix(
            root.to_str().ok_or_else(|| anyhow::anyhow!("utf-8 path"))?,
            false,
        )?;
        assert_eq!(records.len(), 1);
        assert!(records[0].location.ends_with("a.txt"));
        Ok(())
    }

    #[test]
    fn local_index_recursive_includes_nested_files() -> anyhow::Result<()> {
        let root =
            std::env::temp_dir().join(format!("mx8-snapshot-local-rec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("nested"))?;
        std::fs::write(root.join("a.txt"), b"a")?;
        std::fs::write(root.join("nested").join("b.txt"), b"b")?;

        let (records, _canonical) = index_local_prefix(
            root.to_str().ok_or_else(|| anyhow::anyhow!("utf-8 path"))?,
            true,
        )?;
        assert_eq!(records.len(), 2);
        assert!(records.iter().any(|r| r.location.ends_with("a.txt")));
        assert!(records.iter().any(|r| r.location.ends_with("b.txt")));
        Ok(())
    }
}
