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
    pub schema_version: u32,
}

#[derive(Debug, Clone)]
pub struct SnapshotResolverConfig {
    pub lock_stale_after: Duration,
    pub wait_timeout: Duration,
    pub poll_interval: Duration,
    pub dev_manifest_path: Option<PathBuf>,
}

impl Default for SnapshotResolverConfig {
    fn default() -> Self {
        Self {
            lock_stale_after: Duration::from_secs(60),
            wait_timeout: Duration::from_secs(30),
            poll_interval: Duration::from_millis(100),
            dev_manifest_path: None,
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
                let bytes = self.store.get_manifest_bytes(&hash)?;
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
                let bytes = self.store.get_manifest_bytes(&hash)?;
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

            let (_records, canonical_bytes) = match self.cfg.dev_manifest_path.clone() {
                Some(path) => load_dev_manifest_tsv(&path)?,
                None => index_manifest_for_base(base)?,
            };
            let manifest_hash = ManifestHash(mx8_manifest_store::sha256_hex(&canonical_bytes));

            self.store
                .put_manifest_bytes(&manifest_hash, &canonical_bytes)?;
            self.store
                .set_current_snapshot(&intent_key, &manifest_hash)?;

            info!(
                target: "mx8_proof",
                event = "snapshot_resolved",
                mode = if refresh { "refresh" } else { "create" },
                intent_key = %intent_key,
                manifest_hash = %manifest_hash.0,
                manifest_bytes = canonical_bytes.len() as u64,
                "created snapshot"
            );
            return Ok(ResolvedSnapshot {
                intent_key: Some(intent_key),
                manifest_hash,
                manifest_bytes: canonical_bytes,
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
                let bytes = self.store.get_manifest_bytes(&hash)?;
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

fn index_manifest_for_base(base: &str) -> Result<(Vec<ManifestRecord>, Vec<u8>), SnapshotError> {
    let trimmed = base.trim();
    if trimmed.starts_with("s3://") {
        #[cfg(feature = "s3")]
        {
            return index_s3_prefix(trimmed);
        }
        #[cfg(not(feature = "s3"))]
        {
            return Err(SnapshotError::S3FeatureRequired(trimmed.to_string()));
        }
    }

    Err(SnapshotError::IndexUnsupported(trimmed.to_string()))
}

#[cfg(feature = "s3")]
fn index_s3_prefix(base: &str) -> Result<(Vec<ManifestRecord>, Vec<u8>), SnapshotError> {
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

    fn percent_encode(s: &str) -> String {
        // Minimal percent-encoding for embedding arbitrary labels into decode_hint.
        //
        // We encode everything outside a safe set to avoid introducing separators.
        // (decode_hint must not contain tabs/newlines; we keep it ASCII-stable.)
        let mut out = String::with_capacity(s.len());
        for b in s.as_bytes() {
            match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                    out.push(*b as char)
                }
                _ => out.push_str(&format!("%{:02X}", b)),
            }
        }
        out
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
    let client = block_on(client_from_env())??;
    let label_mode = parse_label_mode()?;

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
        return Ok((records, canonical));
    }

    let keys = block_on(async {
        let mut out: Vec<String> = Vec::new();
        let mut token: Option<String> = None;
        loop {
            let mut req = client.list_objects_v2().bucket(&bucket);
            if !prefix.is_empty() {
                req = req.prefix(&prefix);
            }
            if let Some(t) = token.as_deref() {
                req = req.continuation_token(t);
            }
            let resp = req
                .send()
                .await
                .map_err(|e| SnapshotError::S3Index(format!("list_objects_v2 failed: {e:?}")))?;

            if let Some(contents) = resp.contents {
                for obj in contents {
                    let Some(k) = obj.key else { continue };
                    if k.ends_with('/') {
                        continue;
                    }
                    out.push(k);
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
        Ok::<_, SnapshotError>(out)
    })??;

    if keys.is_empty() {
        warn!(
            target: "mx8_proof",
            event = "snapshot_index_empty",
            base = %base,
            bucket = %bucket,
            prefix = %prefix,
            "s3 index produced zero objects"
        );
    }

    let mut keys = keys;
    keys.sort();

    let mut maybe_labels: Vec<Option<String>> = Vec::with_capacity(keys.len());
    let mut label_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    if label_mode != S3LabelMode::None {
        for k in &keys {
            let label = imagefolder_label_for_key(&prefix, k);
            if let Some(l) = &label {
                label_set.insert(l.clone());
            }
            maybe_labels.push(label);
        }
    } else {
        maybe_labels.resize_with(keys.len(), || None);
    }

    let mut label_map: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    let use_imagefolder_labels = match label_mode {
        S3LabelMode::None => false,
        S3LabelMode::ImageFolder => {
            if maybe_labels.iter().any(|l| l.is_none()) {
                return Err(SnapshotError::S3Index(
                    "MX8_S3_LABEL_MODE=imagefolder requires all keys to match prefix/<label>/<file>".to_string(),
                ));
            }
            true
        }
        S3LabelMode::Auto => {
            // Conservative auto-detection: only enable if every key has a label segment and
            // we observe at least 2 distinct labels (otherwise treat as unlabeled).
            maybe_labels.iter().all(|l| l.is_some()) && label_set.len() >= 2
        }
    };

    if use_imagefolder_labels {
        for (i, label) in label_set.into_iter().enumerate() {
            label_map.insert(label, i as u64);
        }
    }

    let mut records: Vec<ManifestRecord> = Vec::with_capacity(keys.len());
    for (i, (key, label)) in keys.into_iter().zip(maybe_labels.into_iter()).enumerate() {
        let location = format!("s3://{bucket}/{key}");
        if location.contains('\n') || location.contains('\t') {
            return Err(SnapshotError::S3Index(format!(
                "s3 object url contains tabs/newlines (sample_id={i}): {location:?}"
            )));
        }
        let decode_hint = if use_imagefolder_labels {
            let label = label.ok_or_else(|| {
                SnapshotError::S3Index(format!(
                    "internal error: label missing under imagefolder mode (sample_id={i})"
                ))
            })?;
            let label_id = *label_map.get(&label).ok_or_else(|| {
                SnapshotError::S3Index(format!(
                    "internal error: label not present in label map (sample_id={i}, label={label:?})"
                ))
            })?;
            let label_enc = percent_encode(&label);
            Some(format!(
                "mx8:vision:imagefolder;label_id={label_id};label={label_enc}"
            ))
        } else {
            None
        };
        let record = ManifestRecord {
            sample_id: i as u64,
            location,
            byte_offset: None,
            byte_length: None,
            decode_hint,
        };
        record.validate().map_err(|e| {
            SnapshotError::S3Index(format!("indexed record failed validation: {e}"))
        })?;
        records.push(record);
    }

    info!(
        target: "mx8_proof",
        event = "snapshot_indexed",
        base = %base,
        bucket = %bucket,
        prefix = %prefix,
        label_mode = if use_imagefolder_labels { "imagefolder" } else { "none" },
        objects = records.len() as u64,
        "indexed s3 prefix"
    );

    let canonical = canonicalize_manifest_bytes(&records);
    Ok((records, canonical))
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
        // Stable TSV with 5 columns; empty string for None.
        // sample_id<TAB>location<TAB>byte_offset<TAB>byte_length<TAB>decode_hint\n
        out.extend_from_slice(r.sample_id.to_string().as_bytes());
        out.push(b'\t');
        out.extend_from_slice(r.location.as_bytes());
        out.push(b'\t');
        if let Some(v) = r.byte_offset {
            out.extend_from_slice(v.to_string().as_bytes());
        }
        out.push(b'\t');
        if let Some(v) = r.byte_length {
            out.extend_from_slice(v.to_string().as_bytes());
        }
        out.push(b'\t');
        if let Some(h) = &r.decode_hint {
            out.extend_from_slice(h.as_bytes());
        }
        out.push(b'\n');
    }
    out
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
}
