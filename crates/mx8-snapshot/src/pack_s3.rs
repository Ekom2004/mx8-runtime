use anyhow::{Context, Result};
use aws_sdk_s3::primitives::ByteStream;
use futures::StreamExt;
use mx8_core::types::{ManifestRecord, MANIFEST_SCHEMA_VERSION};
use tracing::{info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelMode {
    None,
    ImageFolder,
    Auto,
}

#[derive(Debug, Clone)]
pub struct PackS3Config {
    pub pack_in: String,
    pub pack_out: String,
    pub shard_mb: u64,
    pub label_mode: LabelMode,
    pub require_labels: bool,
    /// Number of concurrent S3 GET requests during packing. Default: 32.
    pub parallel_fetches: usize,
}

#[derive(Debug, Clone)]
pub struct PackS3Result {
    pub samples: u64,
    pub shards: u64,
    pub manifest_key: String,
    pub manifest_hash: String,
    pub labels_key: Option<String>,
    pub labels_hash: Option<String>,
}

fn parse_s3_prefix(url: &str) -> Result<(String, String)> {
    // url: s3://bucket/prefix[/]
    let rest = url
        .strip_prefix("s3://")
        .ok_or_else(|| anyhow::anyhow!("invalid s3 url: {url}"))?;
    let s = rest.trim().trim_matches('/');
    let mut it = s.splitn(2, '/');
    let bucket = it.next().unwrap_or("").trim();
    anyhow::ensure!(!bucket.is_empty(), "invalid s3 url (missing bucket): {url}");
    let prefix = it.next().unwrap_or("").trim_matches('/').to_string();
    Ok((bucket.to_string(), prefix))
}

fn percent_encode(s: &str) -> String {
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
    let p = prefix.trim_matches('/');
    let mut rest = key.trim_start_matches('/');
    if !p.is_empty() {
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

fn infer_ext(key: &str) -> Option<&str> {
    let name = key.rsplit('/').next().unwrap_or(key);
    let (_stem, ext) = name.rsplit_once('.')?;
    if ext.is_empty() || ext.len() > 8 {
        return None;
    }
    Some(ext)
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

fn tar_pad_len(len: u64) -> u64 {
    let rem = len % 512;
    if rem == 0 {
        0
    } else {
        512 - rem
    }
}

fn write_octal_field(dst: &mut [u8], val: u64) {
    let width = dst.len();
    for b in dst.iter_mut() {
        *b = 0;
    }
    let digits = width.saturating_sub(1);
    let s = format!("{:0digits$o}", val, digits = digits);
    let bytes = s.as_bytes();
    let start = digits.saturating_sub(bytes.len());
    dst[start..start + bytes.len()].copy_from_slice(bytes);
    dst[digits] = 0;
}

fn tar_header(name: &str, size: u64) -> Result<[u8; 512]> {
    anyhow::ensure!(
        name.len() <= 100,
        "tar name too long (max 100 bytes): {name:?}"
    );

    let mut h = [0u8; 512];
    h[0..name.len()].copy_from_slice(name.as_bytes());

    write_octal_field(&mut h[100..108], 0o644);
    write_octal_field(&mut h[108..116], 0);
    write_octal_field(&mut h[116..124], 0);
    write_octal_field(&mut h[124..136], size);
    write_octal_field(&mut h[136..148], 0);

    for b in &mut h[148..156] {
        *b = b' ';
    }
    h[156] = b'0';

    h[257..263].copy_from_slice(b"ustar\0");
    h[263..265].copy_from_slice(b"00");

    let checksum: u32 = h.iter().map(|b| *b as u32).sum();
    let chk = format!("{:06o}\0 ", checksum);
    h[148..156].copy_from_slice(chk.as_bytes());
    Ok(h)
}

async fn s3_client_from_env() -> Result<aws_sdk_s3::Client> {
    let cfg = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;

    let endpoint_url: Option<String> = std::env::var("MX8_S3_ENDPOINT_URL").ok();
    let force_path_style = match std::env::var("MX8_S3_FORCE_PATH_STYLE") {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "y" | "on")
        }
        Err(std::env::VarError::NotPresent) => endpoint_url.is_some(),
        Err(e) => anyhow::bail!("read env var MX8_S3_FORCE_PATH_STYLE failed: {e}"),
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

/// Default multipart part size. S3 requires ≥5MB per part (except the final part).
/// 16MB gives good throughput and stays well above the minimum.
/// Override with `MX8_PACK_PART_MB`.
const DEFAULT_PART_MB: usize = 16;

/// Streams a tar shard directly to S3 via multipart upload.
/// Bytes are accumulated in `part_buf`; a new part is flushed whenever the
/// buffer reaches `part_size`. This eliminates the local temp-file requirement
/// and overlaps download and upload, roughly halving effective pack time.
struct MultipartState {
    upload_id: String,
    key: String,
    bucket: String,
    part_buf: Vec<u8>,
    part_size: usize,
    completed_parts: Vec<aws_sdk_s3::types::CompletedPart>,
}

impl MultipartState {
    async fn start(
        client: &aws_sdk_s3::Client,
        bucket: &str,
        key: &str,
        part_size: usize,
    ) -> Result<Self> {
        let resp = client
            .create_multipart_upload()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .with_context(|| format!("create_multipart_upload failed: s3://{bucket}/{key}"))?;
        let upload_id = resp
            .upload_id()
            .ok_or_else(|| anyhow::anyhow!("create_multipart_upload: missing upload_id"))?
            .to_string();
        Ok(Self {
            upload_id,
            key: key.to_string(),
            bucket: bucket.to_string(),
            part_buf: Vec::with_capacity(part_size),
            part_size,
            completed_parts: Vec::new(),
        })
    }

    async fn write(&mut self, client: &aws_sdk_s3::Client, data: &[u8]) -> Result<()> {
        self.part_buf.extend_from_slice(data);
        while self.part_buf.len() >= self.part_size {
            let chunk: Vec<u8> = self.part_buf.drain(..self.part_size).collect();
            self.flush_part(client, chunk).await?;
        }
        Ok(())
    }

    async fn flush_part(&mut self, client: &aws_sdk_s3::Client, data: Vec<u8>) -> Result<()> {
        let part_number = self.completed_parts.len() as i32 + 1;
        anyhow::ensure!(
            part_number <= 10_000,
            "multipart upload exceeded 10,000 parts limit for s3://{}/{}",
            self.bucket,
            self.key
        );
        let resp = client
            .upload_part()
            .bucket(&self.bucket)
            .key(&self.key)
            .upload_id(&self.upload_id)
            .part_number(part_number)
            .body(ByteStream::from(data))
            .send()
            .await
            .with_context(|| {
                format!(
                    "upload_part {} failed: s3://{}/{}",
                    part_number, self.bucket, self.key
                )
            })?;
        let etag = resp.e_tag().unwrap_or("").to_string();
        self.completed_parts.push(
            aws_sdk_s3::types::CompletedPart::builder()
                .part_number(part_number)
                .e_tag(etag)
                .build(),
        );
        Ok(())
    }

    async fn finalize(mut self, client: &aws_sdk_s3::Client) -> Result<()> {
        if !self.part_buf.is_empty() {
            let remaining = std::mem::take(&mut self.part_buf);
            self.flush_part(client, remaining).await?;
        }
        let completed = aws_sdk_s3::types::CompletedMultipartUpload::builder()
            .set_parts(Some(self.completed_parts))
            .build();
        client
            .complete_multipart_upload()
            .bucket(&self.bucket)
            .key(&self.key)
            .upload_id(&self.upload_id)
            .multipart_upload(completed)
            .send()
            .await
            .with_context(|| {
                format!(
                    "complete_multipart_upload failed: s3://{}/{}",
                    self.bucket, self.key
                )
            })?;
        Ok(())
    }

    /// Best-effort abort. Called on error to avoid accumulating incomplete
    /// multipart uploads in S3 (which incur storage charges).
    async fn abort(self, client: &aws_sdk_s3::Client) {
        let _ = client
            .abort_multipart_upload()
            .bucket(&self.bucket)
            .key(&self.key)
            .upload_id(&self.upload_id)
            .send()
            .await;
    }
}

/// Check whether a packed manifest already exists for `s3_url`, and if not,
/// pack the prefix in-place.  This is the one-shot "autopack" path:
/// - First call on an unpacked prefix: packs, then returns.
/// - All subsequent calls: HEAD finds the manifest and returns immediately.
pub async fn autopack_if_needed(s3_url: &str, shard_mb: u64) -> Result<()> {
    let rest = s3_url
        .strip_prefix("s3://")
        .ok_or_else(|| anyhow::anyhow!("autopack: not an s3:// URL: {s3_url}"))?
        .trim_matches('/');
    let mut it = rest.splitn(2, '/');
    let bucket = it.next().unwrap_or("").to_string();
    let prefix = it.next().unwrap_or("").trim_matches('/').to_string();
    let manifest_key = if prefix.is_empty() {
        "_mx8/manifest.tsv".to_string()
    } else {
        format!("{prefix}/_mx8/manifest.tsv")
    };

    let client = s3_client_from_env().await?;
    let manifest_exists = match client
        .head_object()
        .bucket(&bucket)
        .key(&manifest_key)
        .send()
        .await
    {
        Ok(_) => true,
        Err(err) => {
            let is_404 = err
                .raw_response()
                .map(|raw| {
                    let s: u16 = raw.status().into();
                    s == 404
                })
                .unwrap_or(false);
            if is_404 {
                false
            } else {
                anyhow::bail!("autopack: HEAD {manifest_key} failed: {err:?}");
            }
        }
    };

    if manifest_exists {
        info!(
            target: "mx8_proof",
            event = "autopack_skipped",
            s3_url,
            "autopack: manifest already exists, skipping pack"
        );
        return Ok(());
    }

    info!(
        target: "mx8_proof",
        event = "autopack_start",
        s3_url,
        shard_mb,
        "autopack: no manifest found, packing prefix in-place"
    );

    pack_s3(PackS3Config {
        pack_in: s3_url.to_string(),
        pack_out: s3_url.to_string(),
        shard_mb,
        label_mode: LabelMode::Auto,
        require_labels: false,
        parallel_fetches: 128,
    })
    .await?;

    info!(
        target: "mx8_proof",
        event = "autopack_complete",
        s3_url,
        "autopack: packing complete"
    );
    Ok(())
}

pub async fn pack_s3(cfg: PackS3Config) -> Result<PackS3Result> {
    anyhow::ensure!(!cfg.pack_in.trim().is_empty(), "pack_in is required");
    anyhow::ensure!(!cfg.pack_out.trim().is_empty(), "pack_out is required");
    anyhow::ensure!(cfg.shard_mb > 0, "shard_mb must be > 0");

    let in_mode = if cfg.require_labels {
        LabelMode::ImageFolder
    } else {
        cfg.label_mode
    };

    let (in_bucket, in_prefix) = parse_s3_prefix(&cfg.pack_in)?;
    let (out_bucket, out_prefix) = parse_s3_prefix(&cfg.pack_out)?;

    let shard_target_bytes = cfg.shard_mb.saturating_mul(1024).saturating_mul(1024);

    let client = s3_client_from_env().await?;

    if let Err(err) = client.create_bucket().bucket(&out_bucket).send().await {
        warn!(
            ?err,
            bucket = out_bucket.as_str(),
            "create_bucket failed (continuing)"
        );
    }

    info!(
        target: "mx8_proof",
        event = "pack_start",
        pack_in = cfg.pack_in.as_str(),
        pack_out = cfg.pack_out.as_str(),
        shard_mb = cfg.shard_mb,
        label_mode = match in_mode {
            LabelMode::None => "none",
            LabelMode::Auto => "auto",
            LabelMode::ImageFolder => "imagefolder",
        },
        "pack starting"
    );

    let mut objects: Vec<(String, u64)> = Vec::new();
    let mut token: Option<String> = None;
    loop {
        let mut req = client.list_objects_v2().bucket(&in_bucket);
        if !in_prefix.is_empty() {
            req = req.prefix(&in_prefix);
        }
        if let Some(t) = token.as_deref() {
            req = req.continuation_token(t);
        }
        let resp = req.send().await?;
        if let Some(contents) = resp.contents {
            for obj in contents {
                let Some(k) = obj.key else { continue };
                if k.ends_with('/') {
                    continue;
                }
                // Skip already-packed output artifacts so in-place packing
                // (pack_in == pack_out) doesn't re-pack its own shards or index.
                let shards_key = if out_prefix.is_empty() {
                    "shards/".to_string()
                } else {
                    format!("{out_prefix}/shards/")
                };
                let mx8_key = if out_prefix.is_empty() {
                    "_mx8/".to_string()
                } else {
                    format!("{out_prefix}/_mx8/")
                };
                if k.starts_with(&shards_key) || k.starts_with(&mx8_key) {
                    continue;
                }
                let sz: u64 = obj.size.and_then(|v| u64::try_from(v).ok()).unwrap_or(0);
                objects.push((k, sz));
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
    objects.sort_by(|a, b| a.0.cmp(&b.0));
    anyhow::ensure!(!objects.is_empty(), "input prefix produced zero objects");

    let mut maybe_labels: Vec<Option<String>> = Vec::with_capacity(objects.len());
    let mut label_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    if in_mode != LabelMode::None {
        for (k, _) in &objects {
            let label = imagefolder_label_for_key(&in_prefix, k);
            if let Some(l) = &label {
                label_set.insert(l.clone());
            }
            maybe_labels.push(label);
        }
    } else {
        maybe_labels.resize_with(objects.len(), || None);
    }

    let use_labels = match in_mode {
        LabelMode::None => false,
        LabelMode::ImageFolder => {
            anyhow::ensure!(
                maybe_labels.iter().all(|l| l.is_some()),
                "require_labels/imagefolder mode requires keys under prefix/<label>/<file>"
            );
            true
        }
        LabelMode::Auto => maybe_labels.iter().all(|l| l.is_some()) && label_set.len() >= 2,
    };

    let labeled = maybe_labels.iter().filter(|l| l.is_some()).count() as u64;
    let unlabeled = (maybe_labels.len() as u64).saturating_sub(labeled);
    if in_mode == LabelMode::Auto && labeled > 0 && unlabeled > 0 {
        warn!(
            target: "mx8_proof",
            event = "pack_label_mixed_layout",
            labeled,
            unlabeled,
            "some keys under the input prefix match ImageFolder, but others do not; disabling labels (set MX8_S3_LABEL_MODE=imagefolder to require)"
        );
    }

    let mut label_map: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    let mut labels_by_id: Vec<String> = Vec::new();
    if use_labels {
        for (i, label) in label_set.into_iter().enumerate() {
            labels_by_id.push(label.clone());
            label_map.insert(label, i as u64);
        }
    }

    let parallel_fetches = {
        let from_env = std::env::var("MX8_PACK_PARALLEL_FETCHES")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok());
        from_env.unwrap_or(cfg.parallel_fetches).max(1)
    };
    let part_size = std::env::var("MX8_PACK_PART_MB")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_PART_MB)
        .max(5) // S3 minimum is 5MB per part
        .saturating_mul(1024 * 1024);

    // Download objects in parallel (ordered) then write sequentially into shards.
    // futures::stream::buffered() drives up to `parallel_fetches` futures concurrently
    // while preserving the original ordering — safe for deterministic byte offsets.
    struct FetchedObject {
        index: usize,
        key: String,
        bytes: Vec<u8>,
        size: u64,
        label: Option<String>,
    }

    let fetch_stream = futures::stream::iter(
        objects
            .iter()
            .cloned()
            .zip(maybe_labels.into_iter())
            .enumerate()
            .map(|(i, ((key, mut size), label))| {
                let client = client.clone();
                let in_bucket = in_bucket.clone();
                async move {
                    if size == 0 {
                        let head = client
                            .head_object()
                            .bucket(&in_bucket)
                            .key(&key)
                            .send()
                            .await?;
                        size = head.content_length().unwrap_or(0) as u64;
                    }
                    anyhow::ensure!(size > 0, "zero/unknown object size: s3://{in_bucket}/{key}");
                    let obj = client
                        .get_object()
                        .bucket(&in_bucket)
                        .key(&key)
                        .send()
                        .await?;
                    let collected = obj.body.collect().await?;
                    let b: Vec<u8> = collected.into_bytes().to_vec();
                    anyhow::ensure!(
                        b.len() as u64 == size,
                        "get_object size mismatch for s3://{}/{} (got {} expected {})",
                        in_bucket,
                        key,
                        b.len(),
                        size,
                    );
                    anyhow::Ok(FetchedObject {
                        index: i,
                        key,
                        bytes: b,
                        size,
                        label,
                    })
                }
            }),
    )
    .buffered(parallel_fetches);

    let mut records: Vec<ManifestRecord> = Vec::with_capacity(objects.len());

    let mut shard_idx: u64 = 0;
    let mut shard_bytes: u64 = 0;
    let mut shard_offset: u64 = 0;
    let mut total_shards: u64 = 0;
    let mut wrote_any_in_shard = false;
    let total_objects = objects.len();
    let pack_started = std::time::Instant::now();
    let mut last_progress = std::time::Instant::now();
    let progress_interval = std::time::Duration::from_secs(2);

    // Helper: build the S3 key for the current shard index.
    let shard_out_key = |idx: u64| -> String {
        if out_prefix.is_empty() {
            format!("shards/shard-{idx:05}.tar")
        } else {
            format!("{out_prefix}/shards/shard-{idx:05}.tar")
        }
    };

    let mut shard =
        MultipartState::start(&client, &out_bucket, &shard_out_key(shard_idx), part_size).await?;

    // Run the main pack loop, aborting the active multipart upload on any error
    // so incomplete uploads don't accumulate in S3.
    let pack_result: Result<()> = async {
        let mut fetch_stream = std::pin::pin!(fetch_stream);
        while let Some(fetched) = fetch_stream.next().await {
            let FetchedObject {
                index: i,
                key,
                bytes: b,
                size,
                label,
            } = fetched?;

            // Print a progress line to stderr every 2 seconds so the user
            // knows the pack is running and hasn't hung.
            if last_progress.elapsed() >= progress_interval {
                let done = records.len() + 1;
                let elapsed = pack_started.elapsed().as_secs_f64();
                let eta = if done > 0 {
                    let rate = done as f64 / elapsed;
                    let remaining = total_objects.saturating_sub(done);
                    if rate > 0.0 {
                        format!("eta={:.0}s", remaining as f64 / rate)
                    } else {
                        "eta=?".to_string()
                    }
                } else {
                    "eta=?".to_string()
                };
                eprintln!(
                    "[mx8-pack] {done}/{total_objects} files  {total_shards} shards  elapsed={elapsed:.0}s  {eta}"
                );
                last_progress = std::time::Instant::now();
            }

            let ext = infer_ext(&key)
                .map(|e| format!(".{e}"))
                .unwrap_or_else(|| ".bin".to_string());
            let name = format!("{:020}{}", i, ext);
            let header = tar_header(&name, size)?;
            let pad = tar_pad_len(size);
            let entry_bytes = 512u64
                .checked_add(size)
                .and_then(|v| v.checked_add(pad))
                .ok_or_else(|| anyhow::anyhow!("tar size overflow"))?;

            // Rotate shard when this entry would overflow.
            if wrote_any_in_shard && shard_bytes.saturating_add(entry_bytes) > shard_target_bytes {
                // Write tar end-of-archive marker and finalize the multipart upload.
                shard.write(&client, &[0u8; 1024]).await?;
                let out_key = shard_out_key(shard_idx);
                info!(
                    target: "mx8_proof",
                    event = "pack_upload_shard",
                    shard_idx,
                    shard_bytes = shard_offset + 1024,
                    shard_key = out_key.as_str(),
                    "uploading shard"
                );
                shard.finalize(&client).await?;
                total_shards = total_shards.saturating_add(1);

                shard_idx = shard_idx.saturating_add(1);
                shard_bytes = 0;
                shard_offset = 0;
                wrote_any_in_shard = false;

                shard = MultipartState::start(
                    &client,
                    &out_bucket,
                    &shard_out_key(shard_idx),
                    part_size,
                )
                .await?;
            }

            let data_offset = shard_offset
                .checked_add(512)
                .ok_or_else(|| anyhow::anyhow!("offset overflow"))?;

            shard.write(&client, &header).await?;
            shard.write(&client, &b).await?;
            if pad != 0 {
                let zeros = vec![0u8; pad as usize];
                shard.write(&client, &zeros).await?;
            }

            wrote_any_in_shard = true;
            shard_bytes = shard_bytes.saturating_add(entry_bytes);
            shard_offset = shard_offset.saturating_add(entry_bytes);

            let location = format!("s3://{out_bucket}/{}", shard_out_key(shard_idx));

            let decode_hint = if use_labels {
                let label = label.context("internal error: missing label under use_labels=true")?;
                let label_id = *label_map
                    .get(&label)
                    .with_context(|| format!("internal error: label not in map: {label:?}"))?;
                Some(format!("mx8:vision:imagefolder;label_id={label_id}"))
            } else {
                None
            };

            let record = ManifestRecord {
                sample_id: i as u64,
                location,
                byte_offset: Some(data_offset),
                byte_length: Some(size),
                decode_hint,
            };
            record.validate()?;
            records.push(record);
        }

        // Finalize the last shard.
        if wrote_any_in_shard {
            shard.write(&client, &[0u8; 1024]).await?;
            let out_key = shard_out_key(shard_idx);
            info!(
                target: "mx8_proof",
                event = "pack_upload_shard",
                shard_idx,
                shard_bytes = shard_offset + 1024,
                shard_key = out_key.as_str(),
                "uploading shard"
            );
            shard.finalize(&client).await?;
            total_shards = total_shards.saturating_add(1);
        } else {
            shard.abort(&client).await;
        }

        Ok(())
    }
    .await;

    pack_result?;

    let manifest_bytes = canonicalize_manifest_bytes(&records);
    let manifest_hash = mx8_manifest_store::sha256_hex(&manifest_bytes);

    let manifest_key = if out_prefix.is_empty() {
        "_mx8/manifest.tsv".to_string()
    } else {
        format!("{out_prefix}/_mx8/manifest.tsv")
    };
    client
        .put_object()
        .bucket(&out_bucket)
        .key(&manifest_key)
        .body(ByteStream::from(manifest_bytes))
        .send()
        .await
        .with_context(|| format!("put_object failed: s3://{out_bucket}/{manifest_key}"))?;

    let (labels_key, labels_hash) = if use_labels {
        let mut bytes: Vec<u8> = Vec::with_capacity(labels_by_id.len() * 16);
        bytes.extend_from_slice(b"schema_version=1\n");
        for (id, label) in labels_by_id.iter().enumerate() {
            bytes.extend_from_slice(id.to_string().as_bytes());
            bytes.push(b'\t');
            bytes.extend_from_slice(percent_encode(label).as_bytes());
            bytes.push(b'\n');
        }
        let hash = mx8_manifest_store::sha256_hex(&bytes);
        let key = if out_prefix.is_empty() {
            "_mx8/labels.tsv".to_string()
        } else {
            format!("{out_prefix}/_mx8/labels.tsv")
        };
        client
            .put_object()
            .bucket(&out_bucket)
            .key(&key)
            .body(ByteStream::from(bytes))
            .send()
            .await
            .with_context(|| format!("put_object failed: s3://{out_bucket}/{key}"))?;
        (Some(key), Some(hash))
    } else {
        (None, None)
    };

    let elapsed_total = pack_started.elapsed().as_secs_f64();
    eprintln!(
        "[mx8-pack] done  {total_objects} files  {total_shards} shards  elapsed={elapsed_total:.0}s  manifest={manifest_key}"
    );

    info!(
        target: "mx8_proof",
        event = "pack_complete",
        pack_out = cfg.pack_out.as_str(),
        samples = records.len() as u64,
        shards = total_shards,
        manifest_key = manifest_key.as_str(),
        manifest_hash = manifest_hash.as_str(),
        "pack complete"
    );

    Ok(PackS3Result {
        samples: records.len() as u64,
        shards: total_shards,
        manifest_key,
        manifest_hash,
        labels_key,
        labels_hash,
    })
}
