use std::collections::BTreeMap;

use mx8_core::types::{ManifestHash, ManifestRecord};
use thiserror::Error;

pub const VIDEO_CLIP_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoStage1Config {
    pub clip_len: u32,
    pub stride: u32,
    pub fps_policy: String,
    pub seed: u64,
    pub epoch: u64,
    pub max_clips_in_memory: usize,
}

impl VideoStage1Config {
    pub fn validate(&self) -> Result<(), VideoStage1Error> {
        if self.clip_len == 0 {
            return Err(VideoStage1Error::InvalidConfig(
                "clip_len must be > 0".to_string(),
            ));
        }
        if self.stride == 0 {
            return Err(VideoStage1Error::InvalidConfig(
                "stride must be > 0".to_string(),
            ));
        }
        if self.fps_policy.trim().is_empty() {
            return Err(VideoStage1Error::InvalidConfig(
                "fps_policy must be non-empty".to_string(),
            ));
        }
        if self.max_clips_in_memory == 0 {
            return Err(VideoStage1Error::InvalidConfig(
                "max_clips_in_memory must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoClipRecord {
    pub sample_id: u64,
    pub media_uri: String,
    pub stream_id: u32,
    pub clip_start: u64,
    pub clip_len: u32,
    pub fps_policy: String,
    pub clip_id: String,
    pub schema_version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoFailureCounts {
    pub corrupt_media: u64,
    pub short_media: u64,
    pub unsupported_codec: u64,
    pub missing_stream: u64,
}

impl VideoFailureCounts {
    fn bump(&mut self, reason: &str) {
        match reason {
            "corrupt_media" => self.corrupt_media = self.corrupt_media.saturating_add(1),
            "short_media" => self.short_media = self.short_media.saturating_add(1),
            "unsupported_codec" => {
                self.unsupported_codec = self.unsupported_codec.saturating_add(1)
            }
            "missing_stream" => self.missing_stream = self.missing_stream.saturating_add(1),
            _ => {}
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoStage1Summary {
    pub clip_count: u64,
    pub tail_clips_dropped_total: u64,
    pub failure_counts: VideoFailureCounts,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoStage1Index {
    pub clips: Vec<VideoClipRecord>,
    pub summary: VideoStage1Summary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedVideoHint {
    frames: u64,
    stream_id: u32,
}

#[derive(Debug, Error)]
pub enum VideoStage1Error {
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("manifest parse error: {0}")]
    ManifestParse(String),
    #[error("schema parse error: {0}")]
    SchemaParse(String),
    #[error("memory bound exceeded while building clip index: built={built} cap={cap}")]
    MemoryBoundExceeded { built: usize, cap: usize },
}

pub fn build_video_stage1_index_from_manifest_bytes(
    manifest_hash: &ManifestHash,
    manifest_bytes: &[u8],
    cfg: &VideoStage1Config,
) -> Result<VideoStage1Index, VideoStage1Error> {
    let records = parse_canonical_manifest_tsv_bytes(manifest_bytes)?;
    build_video_stage1_index(manifest_hash, &records, cfg)
}

pub fn build_video_stage1_index(
    manifest_hash: &ManifestHash,
    records: &[ManifestRecord],
    cfg: &VideoStage1Config,
) -> Result<VideoStage1Index, VideoStage1Error> {
    cfg.validate()?;
    let mut clips = Vec::new();
    let mut next_sample_id: u64 = 0;
    let mut tail_clips_dropped_total: u64 = 0;
    let mut failure_counts = VideoFailureCounts {
        corrupt_media: 0,
        short_media: 0,
        unsupported_codec: 0,
        missing_stream: 0,
    };

    for record in records {
        let Some(raw_hint) = record.decode_hint.as_deref() else {
            failure_counts.bump("missing_stream");
            continue;
        };
        let fields = parse_decode_hint_fields(raw_hint);

        if is_true_flag(&fields, "corrupt") {
            failure_counts.bump("corrupt_media");
            continue;
        }
        if fields
            .get("codec")
            .map(|v| v.eq_ignore_ascii_case("unsupported"))
            .unwrap_or(false)
        {
            failure_counts.bump("unsupported_codec");
            continue;
        }

        let parsed = match parse_video_hint(&fields) {
            Ok(v) => v,
            Err(_) => {
                failure_counts.bump("missing_stream");
                continue;
            }
        };
        if parsed.frames < u64::from(cfg.clip_len) {
            failure_counts.bump("short_media");
            continue;
        }

        let clip_len_u64 = u64::from(cfg.clip_len);
        let stride_u64 = u64::from(cfg.stride);
        let max_start = parsed.frames.saturating_sub(clip_len_u64);
        let num_clips = 1 + (max_start / stride_u64);
        let exact_cover = max_start % stride_u64 == 0;
        if !exact_cover {
            tail_clips_dropped_total = tail_clips_dropped_total.saturating_add(1);
        }

        for i in 0..num_clips {
            if clips.len() >= cfg.max_clips_in_memory {
                return Err(VideoStage1Error::MemoryBoundExceeded {
                    built: clips.len(),
                    cap: cfg.max_clips_in_memory,
                });
            }
            let clip_start = i.saturating_mul(stride_u64);
            let clip_id = compute_clip_id(
                manifest_hash,
                record.sample_id,
                &record.location,
                parsed.stream_id,
                clip_start,
                cfg,
            );
            clips.push(VideoClipRecord {
                sample_id: next_sample_id,
                media_uri: record.location.clone(),
                stream_id: parsed.stream_id,
                clip_start,
                clip_len: cfg.clip_len,
                fps_policy: cfg.fps_policy.clone(),
                clip_id,
                schema_version: VIDEO_CLIP_SCHEMA_VERSION,
            });
            next_sample_id = next_sample_id.saturating_add(1);
        }
    }

    Ok(VideoStage1Index {
        summary: VideoStage1Summary {
            clip_count: clips.len() as u64,
            tail_clips_dropped_total,
            failure_counts,
        },
        clips,
    })
}

pub fn canonicalize_video_stage1_tsv(records: &[VideoClipRecord]) -> Vec<u8> {
    let mut out = Vec::with_capacity(records.len() * 96);
    out.extend_from_slice(format!("video_schema_version={VIDEO_CLIP_SCHEMA_VERSION}\n").as_bytes());
    for r in records {
        let line = format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            r.sample_id,
            r.media_uri,
            r.stream_id,
            r.clip_start,
            r.clip_len,
            r.fps_policy,
            r.clip_id
        );
        out.extend_from_slice(line.as_bytes());
    }
    out
}

pub fn parse_video_stage1_tsv(bytes: &[u8]) -> Result<Vec<VideoClipRecord>, VideoStage1Error> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| VideoStage1Error::SchemaParse(format!("tsv is not utf-8: {e}")))?;
    let mut lines = text.lines();
    let first = lines
        .by_ref()
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| VideoStage1Error::SchemaParse("empty video tsv".to_string()))?;
    let Some((k, v)) = first.split_once('=') else {
        return Err(VideoStage1Error::SchemaParse(
            "missing video_schema_version header".to_string(),
        ));
    };
    if k.trim() != "video_schema_version" {
        return Err(VideoStage1Error::SchemaParse(
            "first line must be video_schema_version=<n>".to_string(),
        ));
    }
    let parsed_version: u32 = v
        .trim()
        .parse()
        .map_err(|_| VideoStage1Error::SchemaParse("invalid video_schema_version".to_string()))?;
    if parsed_version != VIDEO_CLIP_SCHEMA_VERSION {
        return Err(VideoStage1Error::SchemaParse(format!(
            "unsupported video_schema_version {parsed_version}"
        )));
    }

    let mut records = Vec::new();
    for (line_no, raw) in lines.enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 7 {
            return Err(VideoStage1Error::SchemaParse(format!(
                "line {}: expected 7 columns",
                line_no + 2
            )));
        }
        let sample_id = cols[0].parse::<u64>().map_err(|_| {
            VideoStage1Error::SchemaParse(format!("line {}: invalid sample_id", line_no + 2))
        })?;
        let media_uri = cols[1].to_string();
        let stream_id = cols[2].parse::<u32>().map_err(|_| {
            VideoStage1Error::SchemaParse(format!("line {}: invalid stream_id", line_no + 2))
        })?;
        let clip_start = cols[3].parse::<u64>().map_err(|_| {
            VideoStage1Error::SchemaParse(format!("line {}: invalid clip_start", line_no + 2))
        })?;
        let clip_len = cols[4].parse::<u32>().map_err(|_| {
            VideoStage1Error::SchemaParse(format!("line {}: invalid clip_len", line_no + 2))
        })?;
        let fps_policy = cols[5].to_string();
        let clip_id = cols[6].to_string();
        records.push(VideoClipRecord {
            sample_id,
            media_uri,
            stream_id,
            clip_start,
            clip_len,
            fps_policy,
            clip_id,
            schema_version: VIDEO_CLIP_SCHEMA_VERSION,
        });
    }
    Ok(records)
}

fn parse_canonical_manifest_tsv_bytes(
    bytes: &[u8],
) -> Result<Vec<ManifestRecord>, VideoStage1Error> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| VideoStage1Error::ManifestParse(format!("manifest is not utf-8: {e}")))?;
    let mut lines = text.lines();
    let first = lines
        .by_ref()
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| VideoStage1Error::ManifestParse("empty manifest".to_string()))?;
    let Some((k, _v)) = first.split_once('=') else {
        return Err(VideoStage1Error::ManifestParse(
            "manifest header missing schema_version".to_string(),
        ));
    };
    if k.trim() != "schema_version" {
        return Err(VideoStage1Error::ManifestParse(
            "manifest header must be schema_version=<n>".to_string(),
        ));
    }

    let mut records = Vec::new();
    for (line_no, raw) in lines.enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 5 {
            return Err(VideoStage1Error::ManifestParse(format!(
                "line {}: expected 5 manifest columns",
                line_no + 2
            )));
        }
        let sample_id = cols[0].parse::<u64>().map_err(|_| {
            VideoStage1Error::ManifestParse(format!("line {}: invalid sample_id", line_no + 2))
        })?;
        let location = cols[1].to_string();
        let byte_offset = if cols[2].is_empty() {
            None
        } else {
            Some(cols[2].parse::<u64>().map_err(|_| {
                VideoStage1Error::ManifestParse(format!(
                    "line {}: invalid byte_offset",
                    line_no + 2
                ))
            })?)
        };
        let byte_length = if cols[3].is_empty() {
            None
        } else {
            Some(cols[3].parse::<u64>().map_err(|_| {
                VideoStage1Error::ManifestParse(format!(
                    "line {}: invalid byte_length",
                    line_no + 2
                ))
            })?)
        };
        let decode_hint = if cols[4].is_empty() {
            None
        } else {
            Some(cols[4].to_string())
        };
        records.push(ManifestRecord {
            sample_id,
            location,
            byte_offset,
            byte_length,
            decode_hint,
        });
    }
    records.sort_by_key(|r| r.sample_id);
    Ok(records)
}

fn parse_video_hint(
    fields: &BTreeMap<String, String>,
) -> Result<ParsedVideoHint, VideoStage1Error> {
    let frames = fields
        .get("frames")
        .ok_or_else(|| VideoStage1Error::ManifestParse("decode_hint missing frames".to_string()))?
        .parse::<u64>()
        .map_err(|_| VideoStage1Error::ManifestParse("decode_hint frames invalid".to_string()))?;
    let stream_id = fields
        .get("stream_id")
        .ok_or_else(|| {
            VideoStage1Error::ManifestParse("decode_hint missing stream_id".to_string())
        })?
        .parse::<u32>()
        .map_err(|_| {
            VideoStage1Error::ManifestParse("decode_hint stream_id invalid".to_string())
        })?;
    Ok(ParsedVideoHint { frames, stream_id })
}

fn parse_decode_hint_fields(raw_hint: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    if !raw_hint.starts_with("mx8:video") {
        return out;
    }
    for token in raw_hint.split(';').skip(1) {
        let Some((k, v)) = token.split_once('=') else {
            continue;
        };
        out.insert(k.trim().to_string(), v.trim().to_string());
    }
    out
}

fn is_true_flag(fields: &BTreeMap<String, String>, key: &str) -> bool {
    fields
        .get(key)
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

fn compute_clip_id(
    manifest_hash: &ManifestHash,
    media_sample_id: u64,
    media_uri: &str,
    stream_id: u32,
    clip_start: u64,
    cfg: &VideoStage1Config,
) -> String {
    let stable = format!(
        "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
        manifest_hash.0,
        media_sample_id,
        media_uri,
        stream_id,
        clip_start,
        cfg.clip_len,
        cfg.stride,
        cfg.fps_policy,
        cfg.seed,
        cfg.epoch
    );
    mx8_manifest_store::sha256_hex(stable.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> VideoStage1Config {
        VideoStage1Config {
            clip_len: 4,
            stride: 2,
            fps_policy: "fixed_fps:8".to_string(),
            seed: 7,
            epoch: 2,
            max_clips_in_memory: 10_000,
        }
    }

    fn record(sample_id: u64, location: &str, decode_hint: &str) -> ManifestRecord {
        ManifestRecord {
            sample_id,
            location: location.to_string(),
            byte_offset: None,
            byte_length: None,
            decode_hint: Some(decode_hint.to_string()),
        }
    }

    #[test]
    fn video_stage1_replay_is_deterministic() {
        let hash = ManifestHash("abc".to_string());
        let records = vec![record(
            0,
            "s3://bucket/v0.mp4",
            "mx8:video;frames=10;stream_id=0;codec=h264",
        )];
        let first = build_video_stage1_index(&hash, &records, &cfg()).expect("index");
        let second = build_video_stage1_index(&hash, &records, &cfg()).expect("index");
        let first_ids = first
            .clips
            .iter()
            .map(|r| r.clip_id.clone())
            .collect::<Vec<_>>();
        let second_ids = second
            .clips
            .iter()
            .map(|r| r.clip_id.clone())
            .collect::<Vec<_>>();
        assert_eq!(first_ids, second_ids);
    }

    #[test]
    fn video_stage1_counts_failure_taxonomy() {
        let hash = ManifestHash("abc".to_string());
        let records = vec![
            record(
                0,
                "s3://bucket/corrupt.mp4",
                "mx8:video;frames=20;stream_id=0;corrupt=true",
            ),
            record(1, "s3://bucket/short.mp4", "mx8:video;frames=2;stream_id=0"),
            record(
                2,
                "s3://bucket/codec.mp4",
                "mx8:video;frames=20;stream_id=0;codec=unsupported",
            ),
            ManifestRecord {
                sample_id: 3,
                location: "s3://bucket/missing.mp4".to_string(),
                byte_offset: None,
                byte_length: None,
                decode_hint: None,
            },
        ];
        let out = build_video_stage1_index(&hash, &records, &cfg()).expect("index");
        assert_eq!(out.summary.failure_counts.corrupt_media, 1);
        assert_eq!(out.summary.failure_counts.short_media, 1);
        assert_eq!(out.summary.failure_counts.unsupported_codec, 1);
        assert_eq!(out.summary.failure_counts.missing_stream, 1);
        assert!(out.clips.is_empty());
    }

    #[test]
    fn video_stage1_tracks_tail_drop() {
        let hash = ManifestHash("abc".to_string());
        let records = vec![record(
            0,
            "s3://bucket/v0.mp4",
            "mx8:video;frames=11;stream_id=0;codec=h264",
        )];
        let out = build_video_stage1_index(&hash, &records, &cfg()).expect("index");
        assert_eq!(out.summary.tail_clips_dropped_total, 1);
        assert_eq!(out.summary.clip_count, 4);
    }

    #[test]
    fn video_stage1_schema_round_trip() {
        let hash = ManifestHash("abc".to_string());
        let records = vec![record(
            0,
            "s3://bucket/v0.mp4",
            "mx8:video;frames=10;stream_id=0;codec=h264",
        )];
        let out = build_video_stage1_index(&hash, &records, &cfg()).expect("index");
        let bytes = canonicalize_video_stage1_tsv(&out.clips);
        let parsed = parse_video_stage1_tsv(&bytes).expect("parse");
        assert_eq!(parsed.len(), out.clips.len());
        assert_eq!(parsed[0].clip_id, out.clips[0].clip_id);
    }

    #[test]
    fn video_stage1_memory_bound_is_enforced() {
        let hash = ManifestHash("abc".to_string());
        let records = vec![record(
            0,
            "s3://bucket/v0.mp4",
            "mx8:video;frames=100;stream_id=0;codec=h264",
        )];
        let mut small = cfg();
        small.max_clips_in_memory = 2;
        let err = build_video_stage1_index(&hash, &records, &small).expect_err("must fail");
        match err {
            VideoStage1Error::MemoryBoundExceeded { .. } => {}
            other => panic!("expected MemoryBoundExceeded, got {other:?}"),
        }
    }
}
