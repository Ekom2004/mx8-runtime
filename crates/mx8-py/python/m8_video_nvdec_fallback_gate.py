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


def main() -> None:
    tmp_root = Path(os.environ["MX8_VIDEO_NVDEC_GATE_TMP_ROOT"])
    store_root = tmp_root / "store"
    store_root.mkdir(parents=True, exist_ok=True)
    data_root = tmp_root / "videos"
    _make_local_videos(data_root)

    os.environ["MX8_VIDEO_STAGE1_INDEX"] = "1"
    os.environ["MX8_VIDEO_STAGE2_BYTES_PER_CLIP"] = os.environ.get(
        "MX8_VIDEO_STAGE2_BYTES_PER_CLIP", "2048"
    )
    os.environ["MX8_VIDEO_DECODE_BACKEND"] = "nvdec"

    loader = mx8.video(
        str(data_root),
        store=str(store_root),
        recursive=True,
        clip=4,
        stride=2,
        fps=8,
        batch=4,
        seed=53,
        epoch=0,
        ram_gb=4,
        profile="balanced",
    )

    batches = 0
    samples = 0
    for _idx, batch in zip(range(16), loader):
        batches += 1
        samples += len(batch.sample_ids)
        if samples >= 6:
            break

    stats = dict(loader.stats())
    if samples <= 0:
        raise RuntimeError("nvdec fallback gate: loader produced no samples")
    if str(stats.get("video_decode_backend")) != "nvdec":
        raise RuntimeError(
            "nvdec fallback gate: expected configured backend=nvdec "
            f"got={stats.get('video_decode_backend')!r}"
        )
    fallback_total = int(stats.get("video_decode_backend_fallback_total", 0))
    if fallback_total <= 0:
        raise RuntimeError(
            "nvdec fallback gate: expected backend fallback activity when nvdec is unavailable"
        )
    if int(stats.get("video_decode_succeeded_clips_total", 0)) <= 0:
        raise RuntimeError("nvdec fallback gate: expected successful decoded clips")

    print("video_nvdec_fallback_gate_summary")
    print(f"  batches: {batches}")
    print(f"  samples: {samples}")
    print(f"  decode_backend: {stats.get('video_decode_backend')}")
    print(f"  fallback_total: {fallback_total}")
    print(
        f"  decode_succeeded_clips_total: {stats.get('video_decode_succeeded_clips_total', 0)}"
    )


if __name__ == "__main__":
    main()
