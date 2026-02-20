import os
import shutil
import subprocess
import time
from pathlib import Path

import mx8


def _ffmpeg_bin() -> str:
    ffmpeg_bin = os.environ.get("MX8_FFMPEG_BIN", "ffmpeg")
    if shutil.which(ffmpeg_bin) is None:
        raise RuntimeError(f"ffmpeg not found on PATH (MX8_FFMPEG_BIN={ffmpeg_bin})")
    return ffmpeg_bin


def _make_mp4(path: Path, color: str, *, fps: int, seconds: int) -> None:
    ffmpeg_bin = _ffmpeg_bin()
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=96x96:r={fps}:d={seconds}",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed creating {path}: {proc.stderr.strip() or proc.stdout.strip()}"
        )


def _build_dataset(root: Path, video_count: int, fps: int, seconds: int) -> None:
    colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "magenta",
        "cyan",
        "orange",
        "purple",
    ]
    root.mkdir(parents=True, exist_ok=True)
    for idx in range(video_count):
        sub = root / f"shard_{idx % 4}"
        name = f"clip_{idx:03d}.mp4"
        _make_mp4(sub / name, colors[idx % len(colors)], fps=fps, seconds=seconds)


def _check_batch_contract(batch, max_inflight_bytes: int) -> tuple[list[str], int]:
    clip_ids = [str(x) for x in batch.clip_ids]
    sample_ids = [int(x) for x in batch.sample_ids]
    media_uris = [str(x) for x in batch.media_uris]
    clip_starts = [int(x) for x in batch.clip_starts]
    offsets = [int(x) for x in batch.offsets]
    payload = bytes(batch.payload)

    if not clip_ids:
        raise RuntimeError("stress gate: empty video batch")
    if len(clip_ids) != len(sample_ids) or len(clip_ids) != len(media_uris):
        raise RuntimeError("stress gate: clip/sample/media length mismatch")
    if len(clip_starts) != len(clip_ids):
        raise RuntimeError("stress gate: clip_starts length mismatch")
    if len(offsets) != len(clip_ids) + 1:
        raise RuntimeError("stress gate: offsets length mismatch")
    if offsets[0] != 0 or offsets[-1] != len(payload):
        raise RuntimeError("stress gate: offsets boundary mismatch")
    for i in range(len(offsets) - 1):
        if offsets[i] > offsets[i + 1]:
            raise RuntimeError("stress gate: offsets not monotonic")

    if batch.layout != "thwc" or batch.dtype != "u8" or batch.colorspace != "rgb24":
        raise RuntimeError(
            f"stress gate: unexpected contract layout/dtype/colorspace "
            f"{batch.layout}/{batch.dtype}/{batch.colorspace}"
        )
    t = int(batch.frames_per_clip)
    h = int(batch.frame_height)
    w = int(batch.frame_width)
    c = int(batch.channels)
    if min(t, h, w, c) <= 0:
        raise RuntimeError("stress gate: invalid contract dimensions")
    strides = [int(x) for x in batch.strides]
    expect_strides = [h * w * c, w * c, c, 1]
    if strides != expect_strides:
        raise RuntimeError(f"stress gate: bad strides got={strides} want={expect_strides}")
    clip_bytes = t * h * w * c
    for i in range(len(offsets) - 1):
        if offsets[i + 1] - offsets[i] != clip_bytes:
            raise RuntimeError("stress gate: per-clip byte span mismatch")

    payload_bytes = len(payload)
    if payload_bytes > max_inflight_bytes:
        raise RuntimeError(
            f"stress gate: payload_bytes={payload_bytes} exceeds max_inflight_bytes={max_inflight_bytes}"
        )
    return clip_ids, payload_bytes


def _run_loader(
    dataset_root: Path,
    store_root: Path,
    *,
    clip_len: int,
    stride: int,
    fps: int,
    batch_size: int,
    max_batches: int,
    seed: int,
    epoch: int,
    max_inflight_bytes: int,
):
    started = time.perf_counter()
    loader = mx8.video(
        str(dataset_root),
        manifest_store_root=str(store_root),
        recursive=True,
        clip_len=clip_len,
        stride=stride,
        fps=fps,
        batch_size_samples=batch_size,
        seed=seed,
        epoch=epoch,
        constraints=mx8.Constraints(
            max_inflight_bytes=max_inflight_bytes,
            max_process_rss_bytes=None,
        ),
    )
    seq = []
    max_payload = 0
    batch_count = 0
    sample_count = 0
    for _ in range(max_batches):
        try:
            batch = next(loader)
        except StopIteration:
            break
        clip_ids, payload_bytes = _check_batch_contract(batch, max_inflight_bytes)
        batch_count += 1
        sample_count += len(clip_ids)
        max_payload = max(max_payload, payload_bytes)
        seq.extend(clip_ids)
    elapsed_s = max(1e-9, time.perf_counter() - started)
    return loader.stats(), seq, batch_count, sample_count, max_payload, elapsed_s


def main() -> None:
    tmp_root = Path(os.environ["MX8_VIDEO_STAGE2B_STRESS_TMP_ROOT"])
    store_root = tmp_root / "store"
    store_root.mkdir(parents=True, exist_ok=True)
    ds_root = tmp_root / "videos"

    video_count = int(os.environ.get("MX8_VIDEO_STAGE2B_STRESS_VIDEO_COUNT", "20"))
    fps = int(os.environ.get("MX8_VIDEO_STAGE2B_STRESS_FPS", "12"))
    seconds = int(os.environ.get("MX8_VIDEO_STAGE2B_STRESS_SECONDS", "3"))
    clip_len = int(os.environ.get("MX8_VIDEO_STAGE2B_STRESS_CLIP_LEN", "8"))
    stride = int(os.environ.get("MX8_VIDEO_STAGE2B_STRESS_STRIDE", "1"))
    batch_size = int(os.environ.get("MX8_VIDEO_STAGE2B_STRESS_BATCH_SIZE", "16"))
    max_batches = int(os.environ.get("MX8_VIDEO_STAGE2B_STRESS_MAX_BATCHES", "64"))
    seed = int(os.environ.get("MX8_VIDEO_STAGE2B_STRESS_SEED", "41"))
    epoch = int(os.environ.get("MX8_VIDEO_STAGE2B_STRESS_EPOCH", "9"))
    max_inflight_bytes = int(
        os.environ.get("MX8_VIDEO_STAGE2B_STRESS_MAX_INFLIGHT_BYTES", str(8 * 1024 * 1024))
    )

    _build_dataset(ds_root, video_count=video_count, fps=fps, seconds=seconds)

    os.environ["MX8_VIDEO_STAGE1_INDEX"] = "1"
    os.environ["MX8_VIDEO_STAGE2_BYTES_PER_CLIP"] = os.environ.get(
        "MX8_VIDEO_STAGE2_BYTES_PER_CLIP", "49152"
    )

    stats1, seq1, batches1, samples1, max_payload1, elapsed_s1 = _run_loader(
        ds_root,
        store_root,
        clip_len=clip_len,
        stride=stride,
        fps=fps,
        batch_size=batch_size,
        max_batches=max_batches,
        seed=seed,
        epoch=epoch,
        max_inflight_bytes=max_inflight_bytes,
    )
    stats2, seq2, batches2, samples2, max_payload2, elapsed_s2 = _run_loader(
        ds_root,
        store_root,
        clip_len=clip_len,
        stride=stride,
        fps=fps,
        batch_size=batch_size,
        max_batches=max_batches,
        seed=seed,
        epoch=epoch,
        max_inflight_bytes=max_inflight_bytes,
    )

    if seq1 != seq2:
        raise RuntimeError("stress gate: replay determinism failed")
    if batches1 <= 0 or samples1 <= 0:
        raise RuntimeError("stress gate: no batches/samples delivered")
    if batches1 != batches2 or samples1 != samples2:
        raise RuntimeError(
            f"stress gate: run mismatch batches/samples ({batches1},{samples1}) vs ({batches2},{samples2})"
        )

    decode_failed = int(stats1.get("video_decode_failed_total", -1))
    if decode_failed != 0:
        raise RuntimeError(f"stress gate: decode failures observed: {decode_failed}")
    attempted = int(stats1.get("video_decode_attempted_clips_total", -1))
    succeeded = int(stats1.get("video_decode_succeeded_clips_total", -1))
    if attempted != succeeded:
        raise RuntimeError(f"stress gate: attempted/succeeded mismatch {attempted}/{succeeded}")
    if int(stats1.get("video_decode_ms_total", 0)) <= 0:
        raise RuntimeError("stress gate: missing decode timing")

    decode_ms_total = int(stats1.get("video_decode_ms_total", 0))
    samples_per_sec = samples1 / elapsed_s1
    decode_ms_per_batch = decode_ms_total / max(1, batches1)
    decode_ms_per_clip = decode_ms_total / max(1, samples1)

    min_samples_per_sec = float(
        os.environ.get("MX8_VIDEO_STAGE2C_MIN_SAMPLES_PER_SEC", "0")
    )
    max_decode_ms_per_batch = float(
        os.environ.get("MX8_VIDEO_STAGE2C_MAX_DECODE_MS_PER_BATCH", "0")
    )
    max_decode_ms_per_clip = float(
        os.environ.get("MX8_VIDEO_STAGE2C_MAX_DECODE_MS_PER_CLIP", "0")
    )
    if min_samples_per_sec > 0 and samples_per_sec < min_samples_per_sec:
        raise RuntimeError(
            f"stress gate: samples_per_sec={samples_per_sec:.3f} below min={min_samples_per_sec:.3f}"
        )
    if max_decode_ms_per_batch > 0 and decode_ms_per_batch > max_decode_ms_per_batch:
        raise RuntimeError(
            f"stress gate: decode_ms_per_batch={decode_ms_per_batch:.3f} above max={max_decode_ms_per_batch:.3f}"
        )
    if max_decode_ms_per_clip > 0 and decode_ms_per_clip > max_decode_ms_per_clip:
        raise RuntimeError(
            f"stress gate: decode_ms_per_clip={decode_ms_per_clip:.3f} above max={max_decode_ms_per_clip:.3f}"
        )

    max_payload = max(max_payload1, max_payload2)
    if max_payload > max_inflight_bytes:
        raise RuntimeError(
            f"stress gate: max_payload={max_payload} exceeds cap={max_inflight_bytes}"
        )

    print("video_stage2b_stress_gate_summary")
    print("  videos:", video_count)
    print("  batches:", batches1)
    print("  samples:", samples1)
    print("  max_payload_bytes:", max_payload)
    print("  max_inflight_bytes:", max_inflight_bytes)
    print("  run1_elapsed_s:", f"{elapsed_s1:.3f}")
    print("  run2_elapsed_s:", f"{elapsed_s2:.3f}")
    print("  samples_per_sec:", f"{samples_per_sec:.3f}")
    print("  decode_ms_total:", decode_ms_total)
    print("  decode_ms_per_batch:", f"{decode_ms_per_batch:.3f}")
    print("  decode_ms_per_clip:", f"{decode_ms_per_clip:.3f}")
    print("  perf_min_samples_per_sec:", f"{min_samples_per_sec:.3f}")
    print("  perf_max_decode_ms_per_batch:", f"{max_decode_ms_per_batch:.3f}")
    print("  perf_max_decode_ms_per_clip:", f"{max_decode_ms_per_clip:.3f}")
    print("  replay_determinism: ok")
    print("  decode_failures: 0")


if __name__ == "__main__":
    main()
