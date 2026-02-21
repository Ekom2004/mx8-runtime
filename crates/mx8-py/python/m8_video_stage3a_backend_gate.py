import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import mx8


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


def _make_local_videos(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _make_mp4(root / "clip_a.mp4", "red")
    _make_mp4(root / "clip_b.mp4", "green")
    _make_mp4(root / "nested" / "clip_c.mp4", "blue")


def _run_backend(
    backend: str,
    data_root: Path,
    store_root: Path,
    *,
    clip_len: int,
    stride: int,
    fps: int,
    batch_size: int,
    max_batches: int,
    seed: int,
    epoch: int,
):
    os.environ["MX8_VIDEO_DECODE_BACKEND"] = backend
    loader = mx8.video(
        str(data_root),
        manifest_store=str(store_root),
        recursive=True,
        clip_len=clip_len,
        stride=stride,
        fps=fps,
        batch_size_samples=batch_size,
        seed=seed,
        epoch=epoch,
        constraints=mx8.Constraints(
            max_inflight_bytes=128 * 1024 * 1024,
            max_ram_bytes=None,
        ),
    )

    clip_ids = []
    payload_sha = hashlib.sha256()
    batch_count = 0
    sample_count = 0
    for _idx, batch in zip(range(max_batches), loader):
        batch_count += 1
        batch_clip_ids = list(batch.clip_ids)
        batch_sample_ids = list(batch.sample_ids)
        offsets = [int(x) for x in batch.offsets]
        payload = bytes(batch.payload)
        if not batch_clip_ids:
            raise RuntimeError(f"{backend}: batch missing clip_ids")
        if len(batch_sample_ids) != len(batch_clip_ids):
            raise RuntimeError(f"{backend}: sample_ids/clip_ids mismatch")
        if offsets[-1] != len(payload):
            raise RuntimeError(f"{backend}: offsets tail != payload length")
        clip_ids.extend(batch_clip_ids)
        sample_count += len(batch_sample_ids)
        payload_sha.update(payload)

    stats = loader.stats()
    if str(stats.get("video_decode_backend")) != backend:
        raise RuntimeError(
            f"{backend}: stats backend mismatch got={stats.get('video_decode_backend')!r}"
        )
    if int(stats.get("video_decode_succeeded_clips_total", 0)) <= 0:
        raise RuntimeError(f"{backend}: no succeeded clips in stats")

    return {
        "clip_ids": clip_ids,
        "batches": batch_count,
        "samples": sample_count,
        "payload_digest": payload_sha.hexdigest(),
        "stats": dict(stats),
    }


def main() -> None:
    tmp_root = Path(os.environ["MX8_VIDEO_STAGE3A_TMP_ROOT"])
    store_root = tmp_root / "store"
    store_root.mkdir(parents=True, exist_ok=True)
    data_root = tmp_root / "videos"
    _make_local_videos(data_root)

    os.environ["MX8_VIDEO_STAGE1_INDEX"] = "1"
    os.environ["MX8_VIDEO_STAGE2_BYTES_PER_CLIP"] = os.environ.get(
        "MX8_VIDEO_STAGE2_BYTES_PER_CLIP", "2048"
    )

    clip_len = int(os.environ.get("MX8_VIDEO_STAGE3A_CLIP_LEN", "4"))
    stride = int(os.environ.get("MX8_VIDEO_STAGE3A_STRIDE", "2"))
    fps = int(os.environ.get("MX8_VIDEO_STAGE3A_FPS", "8"))
    batch_size = int(os.environ.get("MX8_VIDEO_STAGE3A_BATCH_SIZE", "4"))
    max_batches = int(os.environ.get("MX8_VIDEO_STAGE3A_MAX_BATCHES", "16"))
    seed = int(os.environ.get("MX8_VIDEO_STAGE3A_SEED", "43"))
    epoch = int(os.environ.get("MX8_VIDEO_STAGE3A_EPOCH", "2"))

    cli = _run_backend(
        "cli",
        data_root,
        store_root,
        clip_len=clip_len,
        stride=stride,
        fps=fps,
        batch_size=batch_size,
        max_batches=max_batches,
        seed=seed,
        epoch=epoch,
    )
    ffi = _run_backend(
        "ffi",
        data_root,
        store_root,
        clip_len=clip_len,
        stride=stride,
        fps=fps,
        batch_size=batch_size,
        max_batches=max_batches,
        seed=seed,
        epoch=epoch,
    )

    if cli["clip_ids"] != ffi["clip_ids"]:
        raise RuntimeError("stage3a parity failed: clip_id sequence mismatch (cli vs ffi)")
    if cli["samples"] != ffi["samples"]:
        raise RuntimeError("stage3a parity failed: delivered sample mismatch (cli vs ffi)")

    print("video_stage3a_backend_gate_summary")
    print(f"  clip_ids: {len(cli['clip_ids'])}")
    print(f"  batches_cli: {cli['batches']}")
    print(f"  batches_ffi: {ffi['batches']}")
    print(f"  samples_cli: {cli['samples']}")
    print(f"  samples_ffi: {ffi['samples']}")
    print(f"  payload_digest_cli: {cli['payload_digest']}")
    print(f"  payload_digest_ffi: {ffi['payload_digest']}")
    print("  parity_clip_ids: ok")
    print("  parity_contract: ok")


if __name__ == "__main__":
    main()
