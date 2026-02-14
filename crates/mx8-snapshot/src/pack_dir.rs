#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use mx8_core::types::{ManifestRecord, MANIFEST_SCHEMA_VERSION};
use tracing::{info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelMode {
    None,
    ImageFolder,
    Auto,
}

#[derive(Debug, Clone)]
pub struct PackDirConfig {
    pub in_dir: PathBuf,
    pub out_dir: PathBuf,
    pub shard_mb: u64,
    pub label_mode: LabelMode,
    pub require_labels: bool,
}

#[derive(Debug, Clone)]
pub struct PackDirResult {
    pub samples: u64,
    pub shards: u64,
    pub manifest_path: PathBuf,
    pub manifest_hash: String,
    pub labels_path: Option<PathBuf>,
    pub labels_hash: Option<String>,
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

fn canonicalize_manifest_bytes(records: &[ManifestRecord]) -> Vec<u8> {
    let mut out = Vec::with_capacity(records.len() * 64);
    out.extend_from_slice(format!("schema_version={MANIFEST_SCHEMA_VERSION}\n").as_bytes());
    for r in records {
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

fn walk_files(root: &Path) -> Result<Vec<(PathBuf, String)>> {
    let mut out: Vec<(PathBuf, String)> = Vec::new();
    let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)
            .with_context(|| format!("read_dir failed: {}", dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == "_mx8" || name == "shards" {
                continue;
            }
            let meta = entry.metadata()?;
            if meta.is_dir() {
                stack.push(path);
            } else if meta.is_file() {
                let rel = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                out.push((path, rel));
            }
        }
    }
    out.sort_by(|a, b| a.1.cmp(&b.1));
    Ok(out)
}

fn imagefolder_label_for_rel(rel: &str) -> Option<String> {
    let rel = rel.trim_matches('/');
    let (label, _tail) = rel.split_once('/')?;
    let label = label.trim();
    if label.is_empty() {
        return None;
    }
    Some(label.to_string())
}

fn infer_ext(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_string_lossy().to_string();
    let (_stem, ext) = name.rsplit_once('.')?;
    if ext.is_empty() || ext.len() > 8 {
        return None;
    }
    Some(ext.to_string())
}

pub fn pack_dir(cfg: PackDirConfig) -> Result<PackDirResult> {
    anyhow::ensure!(cfg.shard_mb > 0, "shard_mb must be > 0");
    anyhow::ensure!(cfg.in_dir.is_dir(), "in_dir must be a directory");

    let in_mode = if cfg.require_labels {
        LabelMode::ImageFolder
    } else {
        cfg.label_mode
    };

    let shard_target_bytes = cfg.shard_mb.saturating_mul(1024).saturating_mul(1024);
    let files = walk_files(&cfg.in_dir)?;
    anyhow::ensure!(!files.is_empty(), "input directory produced zero files");

    let mut maybe_labels: Vec<Option<String>> = Vec::with_capacity(files.len());
    let mut label_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    if in_mode != LabelMode::None {
        for (_path, rel) in &files {
            let label = imagefolder_label_for_rel(rel);
            if let Some(l) = &label {
                label_set.insert(l.clone());
            }
            maybe_labels.push(label);
        }
    } else {
        maybe_labels.resize_with(files.len(), || None);
    }

    let use_labels = match in_mode {
        LabelMode::None => false,
        LabelMode::ImageFolder => {
            anyhow::ensure!(
                maybe_labels.iter().all(|l| l.is_some()),
                "require_labels/imagefolder mode requires files under <label>/<file>"
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
            "some files match ImageFolder, but others do not; disabling labels (set label_mode=imagefolder to require)"
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

    std::fs::create_dir_all(cfg.out_dir.join("shards"))?;
    std::fs::create_dir_all(cfg.out_dir.join("_mx8"))?;

    info!(
        target: "mx8_proof",
        event = "pack_start",
        pack_in = %cfg.in_dir.display(),
        pack_out = %cfg.out_dir.display(),
        shard_mb = cfg.shard_mb,
        label_mode = match in_mode {
            LabelMode::None => "none",
            LabelMode::Auto => "auto",
            LabelMode::ImageFolder => "imagefolder",
        },
        "pack starting"
    );

    let mut records: Vec<ManifestRecord> = Vec::with_capacity(files.len());

    let mut shard_idx: u64 = 0;
    let mut shard_bytes: u64 = 0;
    let mut shard_offset: u64 = 0;
    let mut wrote_any_in_shard = false;

    let mut shard_path = cfg
        .out_dir
        .join("shards")
        .join(format!("shard-{shard_idx:05}.tar"));
    let mut shard_file = File::create(&shard_path)?;

    let mut total_shards: u64 = 0;

    for (i, ((path, _rel), label)) in files.into_iter().zip(maybe_labels.into_iter()).enumerate() {
        let size = std::fs::metadata(&path)?.len();
        anyhow::ensure!(size > 0, "zero-length file: {}", path.display());

        let ext = infer_ext(&path)
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
            shard_file.write_all(&[0u8; 1024])?;
            shard_file.flush()?;
            total_shards = total_shards.saturating_add(1);

            shard_idx = shard_idx.saturating_add(1);
            shard_bytes = 0;
            shard_offset = 0;

            shard_path = cfg
                .out_dir
                .join("shards")
                .join(format!("shard-{shard_idx:05}.tar"));
            shard_file = File::create(&shard_path)?;
        }

        let data_offset = shard_offset
            .checked_add(512)
            .ok_or_else(|| anyhow::anyhow!("offset overflow"))?;

        shard_file.write_all(&header)?;

        let mut f = File::open(&path)?;
        let mut buf = [0u8; 64 * 1024];
        let mut wrote: u64 = 0;
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            shard_file.write_all(&buf[..n])?;
            wrote = wrote.saturating_add(n as u64);
        }
        anyhow::ensure!(
            wrote == size,
            "file size mismatch for {} (wrote {} expected {})",
            path.display(),
            wrote,
            size
        );

        if pad != 0 {
            let zeros = vec![0u8; pad as usize];
            shard_file.write_all(&zeros)?;
        }

        wrote_any_in_shard = true;
        shard_bytes = shard_bytes.saturating_add(entry_bytes);
        shard_offset = shard_offset.saturating_add(entry_bytes);

        let shard_abs = shard_path.canonicalize().unwrap_or(shard_path.clone());
        let location = shard_abs.display().to_string();

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
        shard_file.write_all(&[0u8; 1024])?;
        shard_file.flush()?;
        total_shards = total_shards.saturating_add(1);
    }

    let manifest_bytes = canonicalize_manifest_bytes(&records);
    let manifest_hash = mx8_manifest_store::sha256_hex(&manifest_bytes);

    let manifest_path = cfg.out_dir.join("_mx8").join("manifest.tsv");
    std::fs::write(&manifest_path, &manifest_bytes)?;

    let (labels_path, labels_hash) = if use_labels {
        let mut bytes: Vec<u8> = Vec::with_capacity(labels_by_id.len() * 16);
        bytes.extend_from_slice(b"schema_version=1\n");
        for (id, label) in labels_by_id.iter().enumerate() {
            bytes.extend_from_slice(id.to_string().as_bytes());
            bytes.push(b'\t');
            bytes.extend_from_slice(percent_encode(label).as_bytes());
            bytes.push(b'\n');
        }
        let hash = mx8_manifest_store::sha256_hex(&bytes);
        let labels_path = cfg.out_dir.join("_mx8").join("labels.tsv");
        std::fs::write(&labels_path, &bytes)
            .with_context(|| format!("write labels.tsv failed: {}", labels_path.display()))?;
        (Some(labels_path), Some(hash))
    } else {
        (None, None)
    };

    info!(
        target: "mx8_proof",
        event = "pack_complete",
        pack_out = %cfg.out_dir.display(),
        samples = records.len() as u64,
        shards = total_shards,
        manifest_hash = manifest_hash.as_str(),
        "pack complete"
    );

    Ok(PackDirResult {
        samples: records.len() as u64,
        shards: total_shards,
        manifest_path,
        manifest_hash,
        labels_path,
        labels_hash,
    })
}
