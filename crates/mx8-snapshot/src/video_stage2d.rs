use std::collections::BTreeMap;

use thiserror::Error;

pub const VIDEO_RANGE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoRangeChunk {
    pub chunk_index: u32,
    pub start_ms: u64,
    pub end_ms: u64,
    pub start_byte: u64,
    pub end_byte: u64,
    pub keyframe: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoRangeSidecar {
    pub sample_id: u64,
    pub media_uri: String,
    pub stream_id: u32,
    pub codec: String,
    pub schema_version: u32,
    pub chunks: Vec<VideoRangeChunk>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    pub start_byte: u64,
    pub end_byte: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangePlan {
    pub anchor_ms: u64,
    pub clip_start_ms: u64,
    pub clip_end_ms: u64,
    pub ranges: Vec<ByteRange>,
    pub planned_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangePlannerConfig {
    pub max_ranges: usize,
    pub merge_gap_bytes: u64,
}

impl Default for RangePlannerConfig {
    fn default() -> Self {
        Self {
            max_ranges: 8,
            merge_gap_bytes: 0,
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum VideoStage2dError {
    #[error("invalid sidecar: {0}")]
    InvalidSidecar(String),
    #[error("schema parse error: {0}")]
    SchemaParse(String),
    #[error("range planning error: {0}")]
    Plan(String),
}

impl VideoRangeSidecar {
    pub fn validate(&self) -> Result<(), VideoStage2dError> {
        if self.schema_version != VIDEO_RANGE_SCHEMA_VERSION {
            return Err(VideoStage2dError::InvalidSidecar(format!(
                "unsupported schema_version {}",
                self.schema_version
            )));
        }
        if self.chunks.is_empty() {
            return Err(VideoStage2dError::InvalidSidecar(
                "chunks must be non-empty".to_string(),
            ));
        }
        let mut seen_keyframe = false;
        let mut prev_chunk_index: Option<u32> = None;
        let mut prev_start_ms: Option<u64> = None;
        let mut prev_end_ms: Option<u64> = None;
        let mut prev_start_byte: Option<u64> = None;
        let mut prev_end_byte: Option<u64> = None;
        for chunk in &self.chunks {
            if chunk.end_ms <= chunk.start_ms {
                return Err(VideoStage2dError::InvalidSidecar(format!(
                    "chunk {} has non-positive duration",
                    chunk.chunk_index
                )));
            }
            if chunk.end_byte <= chunk.start_byte {
                return Err(VideoStage2dError::InvalidSidecar(format!(
                    "chunk {} has non-positive byte span",
                    chunk.chunk_index
                )));
            }
            if let Some(v) = prev_chunk_index {
                if chunk.chunk_index <= v {
                    return Err(VideoStage2dError::InvalidSidecar(
                        "chunk_index must be strictly increasing".to_string(),
                    ));
                }
            }
            if let Some(v) = prev_start_ms {
                if chunk.start_ms < v {
                    return Err(VideoStage2dError::InvalidSidecar(
                        "start_ms must be non-decreasing".to_string(),
                    ));
                }
            }
            if let Some(v) = prev_end_ms {
                if chunk.start_ms < v {
                    return Err(VideoStage2dError::InvalidSidecar(
                        "time intervals must be non-overlapping".to_string(),
                    ));
                }
            }
            if let Some(v) = prev_start_byte {
                if chunk.start_byte < v {
                    return Err(VideoStage2dError::InvalidSidecar(
                        "start_byte must be non-decreasing".to_string(),
                    ));
                }
            }
            if let Some(v) = prev_end_byte {
                if chunk.start_byte < v {
                    return Err(VideoStage2dError::InvalidSidecar(
                        "byte intervals must be non-overlapping".to_string(),
                    ));
                }
            }
            if chunk.keyframe {
                seen_keyframe = true;
            }
            prev_chunk_index = Some(chunk.chunk_index);
            prev_start_ms = Some(chunk.start_ms);
            prev_end_ms = Some(chunk.end_ms);
            prev_start_byte = Some(chunk.start_byte);
            prev_end_byte = Some(chunk.end_byte);
        }
        if !seen_keyframe {
            return Err(VideoStage2dError::InvalidSidecar(
                "at least one keyframe chunk is required".to_string(),
            ));
        }
        Ok(())
    }
}

pub fn canonicalize_video_stage2d_tsv(sidecars: &[VideoRangeSidecar]) -> Vec<u8> {
    let mut rows = Vec::new();
    for sidecar in sidecars {
        for chunk in &sidecar.chunks {
            rows.push((
                sidecar.sample_id,
                sidecar.media_uri.as_str(),
                sidecar.stream_id,
                sidecar.codec.as_str(),
                sidecar.schema_version,
                chunk.chunk_index,
                chunk.start_ms,
                chunk.end_ms,
                chunk.start_byte,
                chunk.end_byte,
                if chunk.keyframe { 1u8 } else { 0u8 },
            ));
        }
    }
    rows.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.cmp(b.1))
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.5.cmp(&b.5))
    });

    let mut out = Vec::with_capacity(rows.len() * 96);
    out.extend_from_slice(
        format!("video_range_schema_version={VIDEO_RANGE_SCHEMA_VERSION}\n").as_bytes(),
    );
    for row in rows {
        let line = format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            row.0, row.1, row.2, row.3, row.4, row.5, row.6, row.7, row.8, row.9, row.10
        );
        out.extend_from_slice(line.as_bytes());
    }
    out
}

pub fn parse_video_stage2d_tsv(bytes: &[u8]) -> Result<Vec<VideoRangeSidecar>, VideoStage2dError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| VideoStage2dError::SchemaParse(format!("tsv is not utf-8: {e}")))?;
    let mut lines = text.lines();
    let first = lines
        .by_ref()
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| VideoStage2dError::SchemaParse("empty sidecar tsv".to_string()))?;
    let Some((k, v)) = first.split_once('=') else {
        return Err(VideoStage2dError::SchemaParse(
            "missing video_range_schema_version header".to_string(),
        ));
    };
    if k.trim() != "video_range_schema_version" {
        return Err(VideoStage2dError::SchemaParse(
            "first line must be video_range_schema_version=<n>".to_string(),
        ));
    }
    let parsed_version = v.trim().parse::<u32>().map_err(|_| {
        VideoStage2dError::SchemaParse("invalid video_range_schema_version".to_string())
    })?;
    if parsed_version != VIDEO_RANGE_SCHEMA_VERSION {
        return Err(VideoStage2dError::SchemaParse(format!(
            "unsupported video_range_schema_version {parsed_version}"
        )));
    }

    let mut grouped: BTreeMap<(u64, String, u32, String, u32), Vec<VideoRangeChunk>> =
        BTreeMap::new();
    for (line_no, raw) in lines.enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 11 {
            return Err(VideoStage2dError::SchemaParse(format!(
                "line {}: expected 11 columns",
                line_no + 2
            )));
        }
        let sample_id = cols[0].parse::<u64>().map_err(|_| {
            VideoStage2dError::SchemaParse(format!("line {}: invalid sample_id", line_no + 2))
        })?;
        let media_uri = cols[1].to_string();
        let stream_id = cols[2].parse::<u32>().map_err(|_| {
            VideoStage2dError::SchemaParse(format!("line {}: invalid stream_id", line_no + 2))
        })?;
        let codec = cols[3].to_string();
        let schema_version = cols[4].parse::<u32>().map_err(|_| {
            VideoStage2dError::SchemaParse(format!("line {}: invalid schema_version", line_no + 2))
        })?;
        let chunk_index = cols[5].parse::<u32>().map_err(|_| {
            VideoStage2dError::SchemaParse(format!("line {}: invalid chunk_index", line_no + 2))
        })?;
        let start_ms = cols[6].parse::<u64>().map_err(|_| {
            VideoStage2dError::SchemaParse(format!("line {}: invalid start_ms", line_no + 2))
        })?;
        let end_ms = cols[7].parse::<u64>().map_err(|_| {
            VideoStage2dError::SchemaParse(format!("line {}: invalid end_ms", line_no + 2))
        })?;
        let start_byte = cols[8].parse::<u64>().map_err(|_| {
            VideoStage2dError::SchemaParse(format!("line {}: invalid start_byte", line_no + 2))
        })?;
        let end_byte = cols[9].parse::<u64>().map_err(|_| {
            VideoStage2dError::SchemaParse(format!("line {}: invalid end_byte", line_no + 2))
        })?;
        let keyframe = match cols[10] {
            "0" => false,
            "1" => true,
            _ => {
                return Err(VideoStage2dError::SchemaParse(format!(
                    "line {}: invalid keyframe flag",
                    line_no + 2
                )))
            }
        };
        grouped
            .entry((sample_id, media_uri, stream_id, codec, schema_version))
            .or_default()
            .push(VideoRangeChunk {
                chunk_index,
                start_ms,
                end_ms,
                start_byte,
                end_byte,
                keyframe,
            });
    }

    let mut out = Vec::with_capacity(grouped.len());
    for ((sample_id, media_uri, stream_id, codec, schema_version), chunks) in grouped {
        let sidecar = VideoRangeSidecar {
            sample_id,
            media_uri,
            stream_id,
            codec,
            schema_version,
            chunks,
        };
        sidecar.validate()?;
        out.push(sidecar);
    }
    Ok(out)
}

pub fn plan_video_ranges(
    sidecar: &VideoRangeSidecar,
    clip_start_ms: u64,
    clip_len_ms: u64,
    cfg: RangePlannerConfig,
) -> Result<RangePlan, VideoStage2dError> {
    sidecar.validate()?;
    if clip_len_ms == 0 {
        return Err(VideoStage2dError::Plan(
            "clip_len_ms must be > 0".to_string(),
        ));
    }
    if cfg.max_ranges == 0 {
        return Err(VideoStage2dError::Plan(
            "max_ranges must be > 0".to_string(),
        ));
    }
    let clip_end_ms = clip_start_ms
        .checked_add(clip_len_ms)
        .ok_or_else(|| VideoStage2dError::Plan("clip_end_ms overflow".to_string()))?;

    let mut anchor_idx = 0usize;
    for (idx, chunk) in sidecar.chunks.iter().enumerate() {
        if chunk.keyframe && chunk.start_ms <= clip_start_ms {
            anchor_idx = idx;
        }
        if chunk.start_ms > clip_start_ms {
            break;
        }
    }
    if !sidecar.chunks[anchor_idx].keyframe {
        let mut fallback = None;
        for (idx, chunk) in sidecar.chunks.iter().enumerate() {
            if chunk.keyframe {
                fallback = Some(idx);
                break;
            }
        }
        let idx = fallback.ok_or_else(|| {
            VideoStage2dError::Plan("no keyframe chunk available for anchor".to_string())
        })?;
        anchor_idx = idx;
    }

    let anchor_ms = sidecar.chunks[anchor_idx].start_ms;
    let mut ranges = Vec::<ByteRange>::new();
    for chunk in sidecar.chunks.iter().skip(anchor_idx) {
        if chunk.start_ms >= clip_end_ms {
            break;
        }
        ranges.push(ByteRange {
            start_byte: chunk.start_byte,
            end_byte: chunk.end_byte,
        });
    }
    if ranges.is_empty() {
        return Err(VideoStage2dError::Plan(
            "no chunk range intersects requested clip window".to_string(),
        ));
    }

    let mut merged = Vec::<ByteRange>::new();
    for range in ranges {
        match merged.last_mut() {
            Some(prev) if range.start_byte <= prev.end_byte.saturating_add(cfg.merge_gap_bytes) => {
                if range.end_byte > prev.end_byte {
                    prev.end_byte = range.end_byte;
                }
            }
            _ => merged.push(range),
        }
    }

    while merged.len() > cfg.max_ranges {
        let mut best_idx = 0usize;
        let mut best_gap = u64::MAX;
        for idx in 0..merged.len().saturating_sub(1) {
            let lhs = merged[idx];
            let rhs = merged[idx + 1];
            let gap = rhs.start_byte.saturating_sub(lhs.end_byte);
            if gap < best_gap {
                best_gap = gap;
                best_idx = idx;
            }
        }
        let left = merged[best_idx];
        let right = merged[best_idx + 1];
        merged[best_idx] = ByteRange {
            start_byte: left.start_byte,
            end_byte: right.end_byte,
        };
        merged.remove(best_idx + 1);
    }

    let mut planned_bytes = 0u64;
    for range in &merged {
        planned_bytes = planned_bytes
            .checked_add(range.end_byte.saturating_sub(range.start_byte))
            .ok_or_else(|| VideoStage2dError::Plan("planned_bytes overflow".to_string()))?;
    }

    Ok(RangePlan {
        anchor_ms,
        clip_start_ms,
        clip_end_ms,
        ranges: merged,
        planned_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_sidecar() -> VideoRangeSidecar {
        VideoRangeSidecar {
            sample_id: 7,
            media_uri: "s3://bucket/video.mp4".to_string(),
            stream_id: 0,
            codec: "h264".to_string(),
            schema_version: VIDEO_RANGE_SCHEMA_VERSION,
            chunks: vec![
                VideoRangeChunk {
                    chunk_index: 0,
                    start_ms: 0,
                    end_ms: 1000,
                    start_byte: 0,
                    end_byte: 100,
                    keyframe: true,
                },
                VideoRangeChunk {
                    chunk_index: 1,
                    start_ms: 1000,
                    end_ms: 2000,
                    start_byte: 100,
                    end_byte: 220,
                    keyframe: false,
                },
                VideoRangeChunk {
                    chunk_index: 2,
                    start_ms: 2000,
                    end_ms: 3000,
                    start_byte: 220,
                    end_byte: 300,
                    keyframe: false,
                },
                VideoRangeChunk {
                    chunk_index: 3,
                    start_ms: 3000,
                    end_ms: 4000,
                    start_byte: 300,
                    end_byte: 360,
                    keyframe: true,
                },
                VideoRangeChunk {
                    chunk_index: 4,
                    start_ms: 4000,
                    end_ms: 5000,
                    start_byte: 360,
                    end_byte: 500,
                    keyframe: false,
                },
            ],
        }
    }

    #[test]
    fn video_stage2d_sidecar_round_trip() {
        let sidecar = sample_sidecar();
        let bytes = canonicalize_video_stage2d_tsv(std::slice::from_ref(&sidecar));
        let parsed = parse_video_stage2d_tsv(&bytes).expect("parse stage2d");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0], sidecar);
    }

    #[test]
    fn video_stage2d_planner_anchors_to_prior_keyframe() {
        let sidecar = sample_sidecar();
        let plan = plan_video_ranges(
            &sidecar,
            2500,
            1000,
            RangePlannerConfig {
                max_ranges: 8,
                merge_gap_bytes: 0,
            },
        )
        .expect("plan");
        assert_eq!(plan.anchor_ms, 0);
        assert_eq!(plan.ranges.len(), 1);
        assert_eq!(
            plan.ranges[0],
            ByteRange {
                start_byte: 0,
                end_byte: 360
            }
        );
    }

    #[test]
    fn video_stage2d_planner_caps_range_count_deterministically() {
        let mut sidecar = sample_sidecar();
        sidecar.chunks = vec![
            VideoRangeChunk {
                chunk_index: 0,
                start_ms: 0,
                end_ms: 100,
                start_byte: 0,
                end_byte: 10,
                keyframe: true,
            },
            VideoRangeChunk {
                chunk_index: 1,
                start_ms: 100,
                end_ms: 200,
                start_byte: 30,
                end_byte: 40,
                keyframe: false,
            },
            VideoRangeChunk {
                chunk_index: 2,
                start_ms: 200,
                end_ms: 300,
                start_byte: 42,
                end_byte: 52,
                keyframe: false,
            },
            VideoRangeChunk {
                chunk_index: 3,
                start_ms: 300,
                end_ms: 400,
                start_byte: 90,
                end_byte: 100,
                keyframe: true,
            },
        ];
        let cfg = RangePlannerConfig {
            max_ranges: 2,
            merge_gap_bytes: 0,
        };
        let p1 = plan_video_ranges(&sidecar, 50, 320, cfg).expect("plan 1");
        let p2 = plan_video_ranges(&sidecar, 50, 320, cfg).expect("plan 2");
        assert_eq!(p1, p2);
        assert_eq!(p1.ranges.len(), 2);
    }

    #[test]
    fn video_stage2d_rejects_unsorted_chunks() {
        let mut sidecar = sample_sidecar();
        sidecar.chunks.swap(1, 2);
        let err = sidecar.validate().expect_err("must reject");
        assert!(matches!(err, VideoStage2dError::InvalidSidecar(_)));
    }
}
