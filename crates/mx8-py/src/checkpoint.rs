use super::*;

pub(crate) struct DataLoaderCheckpointToken {
    pub(crate) manifest_hash: String,
    pub(crate) schema_version: u32,
    pub(crate) epoch: u32,
    pub(crate) next_sample_id: u64,
    pub(crate) end_id: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct VideoLoaderCheckpointToken {
    pub(crate) manifest_hash: String,
    pub(crate) seed: u64,
    pub(crate) epoch: u64,
    pub(crate) clip_len: u32,
    pub(crate) stride: u32,
    pub(crate) fps: u32,
    pub(crate) next_idx: u64,
    pub(crate) clips_total: u64,
    pub(crate) assigned_rank: u32,
    pub(crate) world_size: u32,
}

impl VideoLoaderCheckpointToken {
    pub(crate) fn encode(&self) -> Vec<u8> {
        format!(
            "{DATA_CHECKPOINT_MAGIC}\nkind={VIDEO_CHECKPOINT_KIND}\nmanifest_hash={}\nseed={}\nepoch={}\nclip_len={}\nstride={}\nfps={}\nnext_idx={}\nclips_total={}\nassigned_rank={}\nworld_size={}\n",
            self.manifest_hash,
            self.seed,
            self.epoch,
            self.clip_len,
            self.stride,
            self.fps,
            self.next_idx,
            self.clips_total,
            self.assigned_rank,
            self.world_size,
        )
        .into_bytes()
    }

    pub(crate) fn decode(raw: &[u8]) -> Result<Self> {
        let text = std::str::from_utf8(raw)?;
        let mut lines = text.lines();
        let magic = lines
            .next()
            .ok_or_else(|| anyhow::anyhow!("empty checkpoint token"))?;
        anyhow::ensure!(
            magic == DATA_CHECKPOINT_MAGIC,
            "unsupported checkpoint token format"
        );

        let mut kv = HashMap::<String, String>::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let Some((k, v)) = line.split_once('=') else {
                anyhow::bail!("invalid checkpoint token line: {line}");
            };
            kv.insert(k.trim().to_string(), v.trim().to_string());
        }
        let kind = kv
            .get("kind")
            .ok_or_else(|| anyhow::anyhow!("checkpoint token missing kind"))?;
        anyhow::ensure!(
            kind == VIDEO_CHECKPOINT_KIND,
            "unsupported checkpoint kind {kind}"
        );

        let token = Self {
            manifest_hash: kv
                .get("manifest_hash")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing manifest_hash"))?
                .to_string(),
            seed: kv
                .get("seed")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing seed"))?
                .parse::<u64>()
                .map_err(|_| anyhow::anyhow!("invalid checkpoint seed"))?,
            epoch: kv
                .get("epoch")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing epoch"))?
                .parse::<u64>()
                .map_err(|_| anyhow::anyhow!("invalid checkpoint epoch"))?,
            clip_len: kv
                .get("clip_len")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing clip_len"))?
                .parse::<u32>()
                .map_err(|_| anyhow::anyhow!("invalid checkpoint clip_len"))?,
            stride: kv
                .get("stride")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing stride"))?
                .parse::<u32>()
                .map_err(|_| anyhow::anyhow!("invalid checkpoint stride"))?,
            fps: kv
                .get("fps")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing fps"))?
                .parse::<u32>()
                .map_err(|_| anyhow::anyhow!("invalid checkpoint fps"))?,
            next_idx: kv
                .get("next_idx")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing next_idx"))?
                .parse::<u64>()
                .map_err(|_| anyhow::anyhow!("invalid checkpoint next_idx"))?,
            clips_total: kv
                .get("clips_total")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing clips_total"))?
                .parse::<u64>()
                .map_err(|_| anyhow::anyhow!("invalid checkpoint clips_total"))?,
            assigned_rank: kv
                .get("assigned_rank")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing assigned_rank"))?
                .parse::<u32>()
                .map_err(|_| anyhow::anyhow!("invalid checkpoint assigned_rank"))?,
            world_size: kv
                .get("world_size")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing world_size"))?
                .parse::<u32>()
                .map_err(|_| anyhow::anyhow!("invalid checkpoint world_size"))?,
        };
        anyhow::ensure!(
            token.next_idx <= token.clips_total,
            "invalid checkpoint token range (next_idx > clips_total)"
        );
        anyhow::ensure!(token.world_size >= 1, "invalid checkpoint token world_size");
        Ok(token)
    }
}

pub(crate) fn parse_csv_u64(raw: &str) -> Result<Vec<u64>> {
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }
    raw.split(',')
        .map(|v| {
            v.trim()
                .parse::<u64>()
                .map_err(|_| anyhow::anyhow!("invalid u64 list value: {v}"))
        })
        .collect()
}

pub(crate) fn parse_csv_bool(raw: &str) -> Result<Vec<bool>> {
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }
    raw.split(',')
        .map(|v| match v.trim() {
            "0" => Ok(false),
            "1" => Ok(true),
            _ => Err(anyhow::anyhow!("invalid bool list value: {v}")),
        })
        .collect()
}

pub(crate) fn parse_csv_i128(raw: &str) -> Result<Vec<i128>> {
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }
    raw.split(',')
        .map(|v| {
            v.trim()
                .parse::<i128>()
                .map_err(|_| anyhow::anyhow!("invalid i128 list value: {v}"))
        })
        .collect()
}

pub(crate) fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

pub(crate) fn hex_to_bytes(raw: &str) -> Result<Vec<u8>> {
    let s = raw.trim();
    if !s.len().is_multiple_of(2) {
        anyhow::bail!("invalid hex length");
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let pair = std::str::from_utf8(&bytes[i..i + 2])?;
        let b = u8::from_str_radix(pair, 16).map_err(|_| anyhow::anyhow!("invalid hex byte"))?;
        out.push(b);
        i += 2;
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub(crate) struct MixLoaderCheckpointToken {
    pub(crate) seed: u64,
    pub(crate) epoch: u64,
    pub(crate) source_count: usize,
    pub(crate) schedule_ticks: u64,
    pub(crate) snapshot_emitted_total: u64,
    pub(crate) active: Vec<bool>,
    pub(crate) delivered_batches: Vec<u64>,
    pub(crate) delivered_samples: Vec<u64>,
    pub(crate) delivered_bytes: Vec<u64>,
    pub(crate) starvation_total: Vec<u64>,
    pub(crate) source_exhausted_total: Vec<u64>,
    pub(crate) steps_since_emit: Vec<u64>,
    pub(crate) scheduler_current: Vec<i128>,
    pub(crate) scheduler_tie_break_offset: usize,
    pub(crate) source_checkpoints: Vec<Vec<u8>>,
}

impl MixLoaderCheckpointToken {
    pub(crate) fn encode(&self) -> Vec<u8> {
        let fmt_u64 = |v: &[u64]| {
            v.iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        };
        let fmt_bool = |v: &[bool]| {
            v.iter()
                .map(|b| if *b { "1".to_string() } else { "0".to_string() })
                .collect::<Vec<_>>()
                .join(",")
        };
        let fmt_i128 = |v: &[i128]| {
            v.iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        };
        let source_cp = self
            .source_checkpoints
            .iter()
            .map(|bytes| bytes_to_hex(bytes))
            .collect::<Vec<_>>()
            .join(";");
        format!(
            "{DATA_CHECKPOINT_MAGIC}\nkind={MIX_CHECKPOINT_KIND}\nseed={}\nepoch={}\nsource_count={}\nschedule_ticks={}\nsnapshot_emitted_total={}\nactive={}\ndelivered_batches={}\ndelivered_samples={}\ndelivered_bytes={}\nstarvation_total={}\nsource_exhausted_total={}\nsteps_since_emit={}\nscheduler_current={}\nscheduler_tie_break_offset={}\nsource_checkpoints={}\n",
            self.seed,
            self.epoch,
            self.source_count,
            self.schedule_ticks,
            self.snapshot_emitted_total,
            fmt_bool(&self.active),
            fmt_u64(&self.delivered_batches),
            fmt_u64(&self.delivered_samples),
            fmt_u64(&self.delivered_bytes),
            fmt_u64(&self.starvation_total),
            fmt_u64(&self.source_exhausted_total),
            fmt_u64(&self.steps_since_emit),
            fmt_i128(&self.scheduler_current),
            self.scheduler_tie_break_offset,
            source_cp,
        )
        .into_bytes()
    }

    pub(crate) fn decode(raw: &[u8]) -> Result<Self> {
        let text = std::str::from_utf8(raw)?;
        let mut lines = text.lines();
        let magic = lines
            .next()
            .ok_or_else(|| anyhow::anyhow!("empty checkpoint token"))?;
        anyhow::ensure!(
            magic == DATA_CHECKPOINT_MAGIC,
            "unsupported checkpoint token format"
        );
        let mut kv = HashMap::<String, String>::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let Some((k, v)) = line.split_once('=') else {
                anyhow::bail!("invalid checkpoint token line: {line}");
            };
            kv.insert(k.trim().to_string(), v.trim().to_string());
        }
        let kind = kv
            .get("kind")
            .ok_or_else(|| anyhow::anyhow!("checkpoint token missing kind"))?;
        anyhow::ensure!(
            kind == MIX_CHECKPOINT_KIND,
            "unsupported checkpoint kind {kind}"
        );

        let source_count = kv
            .get("source_count")
            .ok_or_else(|| anyhow::anyhow!("checkpoint token missing source_count"))?
            .parse::<usize>()
            .map_err(|_| anyhow::anyhow!("invalid checkpoint source_count"))?;

        let active = parse_csv_bool(
            kv.get("active")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing active"))?,
        )?;
        let delivered_batches = parse_csv_u64(
            kv.get("delivered_batches")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing delivered_batches"))?,
        )?;
        let delivered_samples = parse_csv_u64(
            kv.get("delivered_samples")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing delivered_samples"))?,
        )?;
        let delivered_bytes = parse_csv_u64(
            kv.get("delivered_bytes")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing delivered_bytes"))?,
        )?;
        let starvation_total = parse_csv_u64(
            kv.get("starvation_total")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing starvation_total"))?,
        )?;
        let source_exhausted_total =
            parse_csv_u64(kv.get("source_exhausted_total").ok_or_else(|| {
                anyhow::anyhow!("checkpoint token missing source_exhausted_total")
            })?)?;
        let steps_since_emit = parse_csv_u64(
            kv.get("steps_since_emit")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing steps_since_emit"))?,
        )?;
        let scheduler_current = parse_csv_i128(
            kv.get("scheduler_current")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing scheduler_current"))?,
        )?;
        let source_checkpoints = kv
            .get("source_checkpoints")
            .ok_or_else(|| anyhow::anyhow!("checkpoint token missing source_checkpoints"))?
            .split(';')
            .filter(|v| !v.trim().is_empty())
            .map(hex_to_bytes)
            .collect::<Result<Vec<_>>>()?;

        let token = Self {
            seed: kv
                .get("seed")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing seed"))?
                .parse::<u64>()
                .map_err(|_| anyhow::anyhow!("invalid checkpoint seed"))?,
            epoch: kv
                .get("epoch")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing epoch"))?
                .parse::<u64>()
                .map_err(|_| anyhow::anyhow!("invalid checkpoint epoch"))?,
            source_count,
            schedule_ticks: kv
                .get("schedule_ticks")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing schedule_ticks"))?
                .parse::<u64>()
                .map_err(|_| anyhow::anyhow!("invalid checkpoint schedule_ticks"))?,
            snapshot_emitted_total: kv
                .get("snapshot_emitted_total")
                .ok_or_else(|| anyhow::anyhow!("checkpoint token missing snapshot_emitted_total"))?
                .parse::<u64>()
                .map_err(|_| anyhow::anyhow!("invalid checkpoint snapshot_emitted_total"))?,
            active,
            delivered_batches,
            delivered_samples,
            delivered_bytes,
            starvation_total,
            source_exhausted_total,
            steps_since_emit,
            scheduler_current,
            scheduler_tie_break_offset: kv
                .get("scheduler_tie_break_offset")
                .ok_or_else(|| {
                    anyhow::anyhow!("checkpoint token missing scheduler_tie_break_offset")
                })?
                .parse::<usize>()
                .map_err(|_| anyhow::anyhow!("invalid checkpoint scheduler_tie_break_offset"))?,
            source_checkpoints,
        };

        let lens_ok = [
            token.active.len(),
            token.delivered_batches.len(),
            token.delivered_samples.len(),
            token.delivered_bytes.len(),
            token.starvation_total.len(),
            token.source_exhausted_total.len(),
            token.steps_since_emit.len(),
            token.scheduler_current.len(),
            token.source_checkpoints.len(),
        ]
        .iter()
        .all(|v| *v == token.source_count);
        anyhow::ensure!(lens_ok, "checkpoint token source vector length mismatch");
        Ok(token)
    }
}

impl DataLoaderCheckpointToken {
    pub(crate) fn encode(&self) -> Vec<u8> {
        format!(
            "{DATA_CHECKPOINT_MAGIC}\nkind={DATA_CHECKPOINT_KIND}\nmanifest_hash={}\nschema_version={}\nepoch={}\nnext_sample_id={}\nend_id={}\n",
            self.manifest_hash,
            self.schema_version,
            self.epoch,
            self.next_sample_id,
            self.end_id
        )
        .into_bytes()
    }

    pub(crate) fn decode(raw: &[u8]) -> Result<Self> {
        let text = std::str::from_utf8(raw)?;
        let mut lines = text.lines();
        let magic = lines
            .next()
            .ok_or_else(|| anyhow::anyhow!("empty checkpoint token"))?;
        anyhow::ensure!(
            magic == DATA_CHECKPOINT_MAGIC,
            "unsupported checkpoint token format"
        );

        let mut kv = HashMap::<String, String>::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let Some((k, v)) = line.split_once('=') else {
                anyhow::bail!("invalid checkpoint token line: {line}");
            };
            kv.insert(k.trim().to_string(), v.trim().to_string());
        }
        let kind = kv
            .get("kind")
            .ok_or_else(|| anyhow::anyhow!("checkpoint token missing kind"))?;
        anyhow::ensure!(
            kind == DATA_CHECKPOINT_KIND,
            "unsupported checkpoint kind {kind}"
        );

        let manifest_hash = kv
            .get("manifest_hash")
            .ok_or_else(|| anyhow::anyhow!("checkpoint token missing manifest_hash"))?
            .to_string();
        let schema_version = kv
            .get("schema_version")
            .ok_or_else(|| anyhow::anyhow!("checkpoint token missing schema_version"))?
            .parse::<u32>()
            .map_err(|_| anyhow::anyhow!("invalid checkpoint schema_version"))?;
        let epoch = kv
            .get("epoch")
            .ok_or_else(|| anyhow::anyhow!("checkpoint token missing epoch"))?
            .parse::<u32>()
            .map_err(|_| anyhow::anyhow!("invalid checkpoint epoch"))?;
        let next_sample_id = kv
            .get("next_sample_id")
            .ok_or_else(|| anyhow::anyhow!("checkpoint token missing next_sample_id"))?
            .parse::<u64>()
            .map_err(|_| anyhow::anyhow!("invalid checkpoint next_sample_id"))?;
        let end_id = kv
            .get("end_id")
            .ok_or_else(|| anyhow::anyhow!("checkpoint token missing end_id"))?
            .parse::<u64>()
            .map_err(|_| anyhow::anyhow!("invalid checkpoint end_id"))?;
        anyhow::ensure!(
            next_sample_id <= end_id,
            "invalid checkpoint token range (next_sample_id > end_id)"
        );
        Ok(Self {
            manifest_hash,
            schema_version,
            epoch,
            next_sample_id,
            end_id,
        })
    }
}

pub(crate) fn manifest_schema_version_and_sample_count(
    manifest_bytes: &[u8],
) -> Result<(u32, u64)> {
    let manifest = std::str::from_utf8(manifest_bytes)?;
    let mut lines = manifest.lines();
    let first = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("manifest header missing schema_version"))?;
    let Some((k, v)) = first.split_once('=') else {
        anyhow::bail!("manifest header must be schema_version=<n>");
    };
    anyhow::ensure!(
        k.trim() == "schema_version",
        "manifest header must be schema_version=<n>"
    );
    let schema_version = v
        .trim()
        .parse::<u32>()
        .map_err(|_| anyhow::anyhow!("invalid schema_version"))?;

    let mut sample_count = 0u64;
    for line in lines {
        if !line.trim().is_empty() {
            sample_count = sample_count.saturating_add(1);
        }
    }
    Ok((schema_version, sample_count))
}
