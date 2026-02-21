import os
import shutil
import subprocess
from pathlib import Path

import mx8


def _write_bytes(path: Path, fill: bytes, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fill * n)


def _make_local_videos(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _make_mp4(root / "clip_a.mp4", "red")
    _make_mp4(root / "clip_b.mp4", "green")
    _make_mp4(root / "nested" / "clip_c.mp4", "blue")


def _ffmpeg_bin() -> str:
    ffmpeg_bin = os.environ.get("MX8_FFMPEG_BIN", "ffmpeg")
    if shutil.which(ffmpeg_bin) is None:
        raise RuntimeError(f"ffmpeg not found on PATH (MX8_FFMPEG_BIN={ffmpeg_bin})")
    return ffmpeg_bin


def _make_mp4(path: Path, color: str, *, fps: int = 8, seconds: int = 2) -> None:
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
        f"color=c={color}:s=64x64:r={fps}:d={seconds}",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed creating {path}: {proc.stderr.strip() or proc.stdout.strip()}"
        )


def _make_failure_probe_video(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    target = root / "missing_soon.mp4"
    _make_mp4(target, "yellow")
    return target


def _check_video_batch_contract(batch) -> None:
    sample_ids = list(batch.sample_ids)
    clip_ids = list(batch.clip_ids)
    media_uris = list(batch.media_uris)
    clip_starts = list(batch.clip_starts)
    offsets = [int(x) for x in batch.offsets]
    payload = bytes(batch.payload)

    if not sample_ids:
        raise RuntimeError("video batch missing sample_ids")
    if not clip_ids:
        raise RuntimeError("video batch missing clip_ids")
    if len(sample_ids) != len(clip_ids):
        raise RuntimeError("video batch sample_ids/clip_ids length mismatch")
    if len(media_uris) != len(clip_ids):
        raise RuntimeError("video batch media_uris/clip_ids length mismatch")
    if len(clip_starts) != len(clip_ids):
        raise RuntimeError("video batch clip_starts/clip_ids length mismatch")
    if len(offsets) != len(clip_ids) + 1:
        raise RuntimeError("video batch offsets length mismatch")
    if offsets[0] != 0:
        raise RuntimeError("video batch offsets must start at 0")
    if offsets[-1] != len(payload):
        raise RuntimeError("video batch offsets tail must equal payload length")
    for i in range(len(offsets) - 1):
        if offsets[i] > offsets[i + 1]:
            raise RuntimeError("video batch offsets must be monotonic")
    frames = int(batch.frames_per_clip)
    height = int(batch.frame_height)
    width = int(batch.frame_width)
    channels = int(batch.channels)
    if frames <= 0 or height <= 0 or width <= 0 or channels <= 0:
        raise RuntimeError("video contract dimensions must be > 0")
    if batch.layout != "thwc":
        raise RuntimeError(f"unexpected layout: {batch.layout!r}")
    if batch.dtype != "u8":
        raise RuntimeError(f"unexpected dtype: {batch.dtype!r}")
    if batch.colorspace != "rgb24":
        raise RuntimeError(f"unexpected colorspace: {batch.colorspace!r}")
    expected_strides = [height * width * channels, width * channels, channels, 1]
    got_strides = [int(x) for x in batch.strides]
    if got_strides != expected_strides:
        raise RuntimeError(
            f"video strides mismatch: got={got_strides} want={expected_strides}"
        )
    expected_clip_bytes = frames * height * width * channels
    for i in range(len(offsets) - 1):
        if offsets[i + 1] - offsets[i] != expected_clip_bytes:
            raise RuntimeError(
                f"clip payload byte size mismatch for clip[{i}] got={offsets[i + 1] - offsets[i]} want={expected_clip_bytes}"
            )


def _collect_sequence(loader, max_batches: int) -> list[str]:
    seq = []
    for _idx, batch in zip(range(max_batches), loader):
        _check_video_batch_contract(batch)
        seq.extend(str(x) for x in batch.clip_ids)
    return seq


def _write_stage1_failure_manifest(path: Path) -> None:
    rows = [
        "0\ts3://bucket/video_ok.mp4\t\t\tmx8:video;frames=17;stream_id=0;codec=h264",
        "1\ts3://bucket/video_corrupt.mp4\t\t\tmx8:video;frames=25;stream_id=0;corrupt=true",
        "2\ts3://bucket/video_short.mp4\t\t\tmx8:video;frames=2;stream_id=0;codec=h264",
        "3\ts3://bucket/video_codec.mp4\t\t\tmx8:video;frames=25;stream_id=0;codec=unsupported",
        "4\ts3://bucket/video_missing_stream.mp4\t\t\tmx8:video;frames=25;codec=h264",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _check_stage1_failure_taxonomy(store_root: Path, tmp_root: Path) -> dict:
    manifest = tmp_root / "video_stage2b_failures.tsv"
    _write_stage1_failure_manifest(manifest)
    out = mx8._internal.video_index_build(
        "s3://bucket/video_stage2b/@refresh",
        manifest_store=str(store_root),
        manifest_path=str(manifest),
        recursive=True,
        clip_len=4,
        stride=2,
        fps_policy="fixed_fps:8",
        seed=11,
        epoch=2,
        max_clips_in_memory=10_000,
    )
    failures = dict(out.get("failure_counts", {}))
    expected = {
        "corrupt_media": 1,
        "short_media": 1,
        "unsupported_codec": 1,
        "missing_stream": 1,
    }
    for key, want in expected.items():
        got = int(failures.get(key, -1))
        if got != want:
            raise RuntimeError(
                f"stage1 failure taxonomy mismatch for {key}: got={got} want={want}"
            )
    return failures


def _check_runtime_io_failure(
    store_root: Path, data_root: Path, clip_len: int, stride: int, fps: int, seed: int, epoch: int
) -> str:
    constraints = mx8.Constraints(
        max_inflight_bytes=8 * 1024 * 1024,
        max_ram_bytes=None,
    )
    _ffmpeg_bin()
    doomed = _make_failure_probe_video(data_root)
    loader = mx8.video(
        str(data_root),
        manifest_store=str(store_root),
        recursive=True,
        clip_len=clip_len,
        stride=stride,
        fps=fps,
        batch_size_samples=1,
        seed=seed,
        epoch=epoch,
        constraints=constraints,
    )
    doomed.unlink()

    try:
        for _ in range(64):
            next(loader)
    except RuntimeError as e:
        msg = str(e)
        if "video decode io_read_failed" in msg:
            return "io_read_failed"
        raise RuntimeError(f"unexpected runtime decode failure: {msg}")
    raise RuntimeError("expected runtime IO decode failure but loader did not fail")


def main() -> None:
    tmp_root = Path(os.environ["MX8_VIDEO_STAGE2B_TMP_ROOT"])
    store_root = tmp_root / "store"
    store_root.mkdir(parents=True, exist_ok=True)
    ds_root = tmp_root / "videos"
    _make_local_videos(ds_root)

    os.environ["MX8_VIDEO_STAGE1_INDEX"] = "1"
    os.environ["MX8_VIDEO_STAGE2_BYTES_PER_CLIP"] = os.environ.get(
        "MX8_VIDEO_STAGE2_BYTES_PER_CLIP", "2048"
    )

    clip_len = int(os.environ.get("MX8_VIDEO_STAGE2B_CLIP_LEN", "4"))
    stride = int(os.environ.get("MX8_VIDEO_STAGE2B_STRIDE", "2"))
    fps = int(os.environ.get("MX8_VIDEO_STAGE2B_FPS", "8"))
    batch_size = int(os.environ.get("MX8_VIDEO_STAGE2B_BATCH_SIZE", "4"))
    max_batches = int(os.environ.get("MX8_VIDEO_STAGE2B_MAX_BATCHES", "12"))
    seed = int(os.environ.get("MX8_VIDEO_STAGE2B_SEED", "31"))
    epoch = int(os.environ.get("MX8_VIDEO_STAGE2B_EPOCH", "7"))

    constraints = mx8.Constraints(
        max_inflight_bytes=128 * 1024,
        max_ram_bytes=None,
    )

    loader1 = mx8.video(
        str(ds_root),
        manifest_store=str(store_root),
        recursive=True,
        clip_len=clip_len,
        stride=stride,
        fps=fps,
        batch_size_samples=batch_size,
        seed=seed,
        epoch=epoch,
        constraints=constraints,
    )
    seq1 = _collect_sequence(loader1, max_batches=max_batches)
    stats1 = loader1.stats()

    loader2 = mx8.video(
        str(ds_root),
        manifest_store=str(store_root),
        recursive=True,
        clip_len=clip_len,
        stride=stride,
        fps=fps,
        batch_size_samples=batch_size,
        seed=seed,
        epoch=epoch,
        constraints=constraints,
    )
    seq2 = _collect_sequence(loader2, max_batches=max_batches)
    if seq1 != seq2:
        raise RuntimeError("video stage2b replay mismatch for same seed/epoch")

    if int(stats1.get("video_delivered_batches_total", 0)) <= 0:
        raise RuntimeError(f"unexpected stats: {stats1}")
    if int(stats1.get("video_delivered_samples_total", 0)) <= 0:
        raise RuntimeError(f"unexpected stats: {stats1}")
    if int(stats1.get("video_delivered_bytes_total", 0)) <= 0:
        raise RuntimeError(f"unexpected stats: {stats1}")
    expected_stats = {
        "video_layout": "thwc",
        "video_dtype": "u8",
        "video_colorspace": "rgb24",
    }
    for key, want in expected_stats.items():
        got = str(stats1.get(key, ""))
        if got != want:
            raise RuntimeError(f"unexpected stat {key}: got={got!r} want={want!r}")
    if int(stats1.get("video_decode_attempted_clips_total", 0)) <= 0:
        raise RuntimeError(f"missing decode attempted clips stat: {stats1}")
    if int(stats1.get("video_decode_succeeded_clips_total", 0)) <= 0:
        raise RuntimeError(f"missing decode succeeded clips stat: {stats1}")
    if int(stats1.get("video_decode_ms_total", 0)) <= 0:
        raise RuntimeError(f"missing decode timing stat: {stats1}")

    try:
        mx8.video(
            str(ds_root),
            manifest_store=str(store_root),
            recursive=True,
            clip_len=clip_len,
            stride=stride,
            fps=fps,
            batch_size_samples=batch_size,
            seed=seed,
            epoch=epoch,
            constraints=mx8.Constraints(
                max_inflight_bytes=2 * 1024,
                max_ram_bytes=None,
            ),
        )
        raise RuntimeError("expected max_inflight_bytes guard to reject config")
    except ValueError as e:
        if "exceeds max_inflight_bytes" not in str(e):
            raise RuntimeError(f"unexpected config error: {e}")

    runtime_failure_class = _check_runtime_io_failure(
        store_root=store_root,
        data_root=tmp_root / "runtime_failure_probe",
        clip_len=clip_len,
        stride=stride,
        fps=fps,
        seed=seed,
        epoch=epoch,
    )
    taxonomy = _check_stage1_failure_taxonomy(store_root=store_root, tmp_root=tmp_root)

    print("video_stage2b_gate_summary")
    print("  batches:", int(stats1["video_delivered_batches_total"]))
    print("  samples:", int(stats1["video_delivered_samples_total"]))
    print("  bytes:", int(stats1["video_delivered_bytes_total"]))
    print("  clips_total:", int(stats1["clips_total"]))
    print("  replay_determinism: ok")
    print("  memory_bound_init_gate: ok")
    print("  runtime_failure_class:", runtime_failure_class)
    print("  stage1_failure_taxonomy:", taxonomy)


if __name__ == "__main__":
    main()
