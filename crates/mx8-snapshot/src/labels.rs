#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use anyhow::{Context, Result};

fn percent_decode(s: &str) -> Result<String> {
    let mut out: Vec<u8> = Vec::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' => {
                anyhow::ensure!(i + 2 < bytes.len(), "bad percent-encoding (truncated)");
                let hi = bytes[i + 1];
                let lo = bytes[i + 2];
                let hex = [hi, lo];
                let v = std::str::from_utf8(&hex)
                    .ok()
                    .and_then(|h| u8::from_str_radix(h, 16).ok())
                    .context("bad percent-encoding")?;
                out.push(v);
                i += 3;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    Ok(String::from_utf8(out)?)
}

pub fn parse_labels_tsv(bytes: &[u8]) -> Result<Vec<String>> {
    let s = std::str::from_utf8(bytes)?;
    let mut lines = s.lines();
    let first = lines
        .by_ref()
        .find(|l| !l.trim().is_empty())
        .context("empty labels.tsv")?;

    let Some((k, v)) = first.split_once('=') else {
        anyhow::bail!("labels header missing schema_version");
    };
    anyhow::ensure!(
        k.trim() == "schema_version",
        "labels header must be schema_version=<n>"
    );
    let schema_version: u32 = v.trim().parse().context("invalid schema_version")?;
    anyhow::ensure!(
        schema_version == 1,
        "unsupported labels schema_version {schema_version}"
    );

    let mut items: Vec<(u64, String)> = Vec::new();
    for (line_no, raw) in lines.enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((id_s, label_enc)) = line.split_once('\t') else {
            anyhow::bail!("labels line {}: expected id<TAB>label", line_no + 2);
        };
        let id: u64 = id_s.trim().parse().context("bad label_id")?;
        let label = percent_decode(label_enc.trim())?;
        items.push((id, label));
    }

    items.sort_by_key(|(id, _)| *id);
    if items.is_empty() {
        return Ok(Vec::new());
    }

    let max_id = items.last().map(|(id, _)| *id).unwrap_or(0);
    let mut out: Vec<Option<String>> = vec![None; (max_id + 1) as usize];
    for (id, label) in items {
        let slot = out.get_mut(id as usize).context("label_id out of bounds")?;
        anyhow::ensure!(slot.is_none(), "duplicate label_id {id}");
        *slot = Some(label);
    }

    let mut labels: Vec<String> = Vec::with_capacity(out.len());
    for (id, v) in out.into_iter().enumerate() {
        labels.push(v.context(format!("missing label_id {id}"))?);
    }
    Ok(labels)
}

pub fn labels_path_for_base(base: &str) -> String {
    let b = base.trim().trim_end_matches('/');
    if b.starts_with("s3://") {
        // `s3://bucket/prefix/_mx8/labels.tsv`
        if b == "s3://" {
            return "s3://_mx8/labels.tsv".to_string();
        }
        format!("{b}/_mx8/labels.tsv")
    } else {
        let p = std::path::PathBuf::from(b);
        p.join("_mx8").join("labels.tsv").display().to_string()
    }
}

pub async fn load_labels_for_base(base: &str) -> Result<Option<Vec<String>>> {
    let b = base.trim().trim_end_matches('/');
    if b.is_empty() {
        return Ok(None);
    }

    if b.starts_with("s3://") {
        #[cfg(not(feature = "s3"))]
        {
            let _ = b;
            return Ok(None);
        }
        #[cfg(feature = "s3")]
        {
            let url = labels_path_for_base(b);
            let rest = url
                .strip_prefix("s3://")
                .context("invalid s3 url for labels")?;
            let mut it = rest.splitn(2, '/');
            let bucket = it.next().unwrap_or("").to_string();
            let key = it.next().unwrap_or("").to_string();
            if bucket.is_empty() || key.is_empty() {
                return Ok(None);
            }

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

            let mut bld = aws_sdk_s3::config::Builder::from(&cfg);
            if let Some(url) = endpoint_url {
                bld = bld.endpoint_url(url);
            }
            if force_path_style {
                bld = bld.force_path_style(true);
            }
            let client = aws_sdk_s3::Client::from_conf(bld.build());

            let resp = client.get_object().bucket(bucket).key(key).send().await;
            let Ok(resp) = resp else {
                return Ok(None);
            };
            let collected = resp.body.collect().await?;
            let bytes = collected.into_bytes();
            let labels = parse_labels_tsv(bytes.as_ref())?;
            return Ok(Some(labels));
        }
    }

    let path = std::path::PathBuf::from(labels_path_for_base(b));
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&path)
        .with_context(|| format!("read labels.tsv failed: {}", path.display()))?;
    let labels = parse_labels_tsv(&bytes)?;
    Ok(Some(labels))
}
