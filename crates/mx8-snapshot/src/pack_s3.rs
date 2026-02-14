use std::path::PathBuf;

use anyhow::{Context, Result};
use aws_sdk_s3::primitives::ByteStream;
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

async fn upload_file(
    client: &aws_sdk_s3::Client,
    bucket: &str,
    key: &str,
    path: &PathBuf,
) -> Result<()> {
    let body = ByteStream::from_path(path)
        .await
        .with_context(|| format!("ByteStream::from_path failed: {}", path.display()))?;

    client
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(body)
        .send()
        .await
        .with_context(|| format!("put_object failed: s3://{bucket}/{key}"))?;

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
                if k.ends_with("/_mx8/manifest.tsv") || k == "_mx8/manifest.tsv" {
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

    let tmp_root = {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "mx8-pack-s3-{}-{}",
            std::process::id(),
            mx8_manifest_store::sha256_hex(cfg.pack_out.as_bytes())
        ));
        std::fs::create_dir_all(&p)?;
        p
    };

    let mut records: Vec<ManifestRecord> = Vec::with_capacity(objects.len());

    let mut shard_idx: u64 = 0;
    let mut shard_bytes: u64 = 0;
    let mut shard_offset: u64 = 0;
    let mut shard_path: PathBuf = tmp_root.join(format!("shard-{shard_idx:05}.tar"));
    let mut shard_file = tokio::fs::File::create(&shard_path).await?;

    let mut total_shards: u64 = 0;

    use tokio::io::AsyncWriteExt;

    let mut wrote_any_in_shard = false;

    for (i, ((key, mut size), label)) in objects
        .iter()
        .cloned()
        .zip(maybe_labels.into_iter())
        .enumerate()
    {
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

        if wrote_any_in_shard && shard_bytes.saturating_add(entry_bytes) > shard_target_bytes {
            shard_file.write_all(&[0u8; 1024]).await?;
            shard_file.flush().await?;

            let out_key = if out_prefix.is_empty() {
                format!("shards/shard-{shard_idx:05}.tar")
            } else {
                format!("{out_prefix}/shards/shard-{shard_idx:05}.tar")
            };
            info!(
                target: "mx8_proof",
                event = "pack_upload_shard",
                shard_idx,
                shard_bytes = shard_offset + 1024,
                shard_key = out_key.as_str(),
                "uploading shard"
            );
            drop(shard_file);
            upload_file(&client, &out_bucket, &out_key, &shard_path).await?;

            total_shards = total_shards.saturating_add(1);

            shard_idx = shard_idx.saturating_add(1);
            shard_bytes = 0;
            shard_offset = 0;

            shard_path = tmp_root.join(format!("shard-{shard_idx:05}.tar"));
            shard_file = tokio::fs::File::create(&shard_path).await?;
        }

        let data_offset = shard_offset
            .checked_add(512)
            .ok_or_else(|| anyhow::anyhow!("offset overflow"))?;

        shard_file.write_all(&header).await?;

        let obj = client
            .get_object()
            .bucket(&in_bucket)
            .key(&key)
            .send()
            .await?;
        let collected = obj.body.collect().await?;
        let b = collected.into_bytes();
        shard_file.write_all(&b).await?;
        let wrote: u64 = b.len() as u64;
        anyhow::ensure!(
            wrote == size,
            "get_object size mismatch for s3://{}/{} (wrote {} expected {})",
            in_bucket,
            key,
            wrote,
            size
        );

        if pad != 0 {
            let zeros = vec![0u8; pad as usize];
            shard_file.write_all(&zeros).await?;
        }

        wrote_any_in_shard = true;
        shard_bytes = shard_bytes.saturating_add(entry_bytes);
        shard_offset = shard_offset.saturating_add(entry_bytes);

        let shard_key = if out_prefix.is_empty() {
            format!("shards/shard-{shard_idx:05}.tar")
        } else {
            format!("{out_prefix}/shards/shard-{shard_idx:05}.tar")
        };
        let location = format!("s3://{out_bucket}/{shard_key}");

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

    if wrote_any_in_shard {
        shard_file.write_all(&[0u8; 1024]).await?;
        shard_file.flush().await?;

        let out_key = if out_prefix.is_empty() {
            format!("shards/shard-{shard_idx:05}.tar")
        } else {
            format!("{out_prefix}/shards/shard-{shard_idx:05}.tar")
        };
        info!(
            target: "mx8_proof",
            event = "pack_upload_shard",
            shard_idx,
            shard_bytes = shard_offset + 1024,
            shard_key = out_key.as_str(),
            "uploading shard"
        );
        drop(shard_file);
        upload_file(&client, &out_bucket, &out_key, &shard_path).await?;

        total_shards = total_shards.saturating_add(1);
    }

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
