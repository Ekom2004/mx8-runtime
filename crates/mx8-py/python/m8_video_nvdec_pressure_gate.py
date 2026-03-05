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
        f"color=c={color}:s=128x128:r={fps}:d={seconds}",
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
    tmp_root = Path(os.environ["MX8_VIDEO_NVDEC_PRESSURE_TMP_ROOT"])
    store_root = tmp_root / "store"
    store_root.mkdir(parents=True, exist_ok=True)
    data_root = tmp_root / "videos"
    _make_local_videos(data_root)

    os.environ["MX8_VIDEO_STAGE1_INDEX"] = "1"
    os.environ["MX8_VIDEO_STAGE2_BYTES_PER_CLIP"] = os.environ.get(
        "MX8_VIDEO_STAGE2_BYTES_PER_CLIP", "4096"
    )
    os.environ["MX8_VIDEO_DECODE_BACKEND"] = "nvdec"
    os.environ["MX8_VIDEO_AUTOTUNE_PERIOD_BATCHES"] = "1"
    # Deterministic stress signal for this gate.
    os.environ["MX8_VIDEO_GPU_PRESSURE_RATIO"] = os.environ.get(
        "MX8_VIDEO_GPU_PRESSURE_RATIO", "0.99"
    )

    batch_size = 4
    loader = mx8.video(
        str(data_root),
        store=str(store_root),
        recursive=True,
        clip=4,
        stride=2,
        fps=8,
        batch=batch_size,
        constraints=mx8.Constraints(max_inflight_bytes=32 * 1024 * 1024),
        ram_gb=4,
        profile="balanced",
        tune=True,
    )

    initial = dict(loader.stats())
    initial_inflight = int(initial.get("max_inflight_bytes", 0))
    if initial_inflight <= 0:
        raise RuntimeError("pressure gate: invalid initial max_inflight_bytes")

    batches = 0
    samples = 0
    for _idx, batch in zip(range(32), loader):
        batches += 1
        samples += len(batch.sample_ids)
        if samples >= 8:
            break

    stats = dict(loader.stats())
    if samples <= 0:
        raise RuntimeError("pressure gate: loader produced no samples")

    final_inflight = int(stats.get("max_inflight_bytes", 0))
    clip_bytes = int(stats.get("video_clip_bytes", 0))
    min_expected = max(1, batch_size * clip_bytes)
    pressure = float(stats.get("video_runtime_autotune_pressure", 0.0))
    gpu_pressure = float(stats.get("video_gpu_pressure", 0.0))
    adjustments_total = int(stats.get("video_runtime_autotune_adjustments_total", 0))
    gpu_clamps_total = int(stats.get("video_runtime_autotune_gpu_clamps_total", 0))

    if pressure < 0.97:
        raise RuntimeError(f"pressure gate: expected pressure >= 0.97, got {pressure}")
    if gpu_pressure < 0.97:
        raise RuntimeError(
            f"pressure gate: expected gpu pressure >= 0.97, got {gpu_pressure}"
        )
    if adjustments_total <= 0:
        raise RuntimeError("pressure gate: expected autotune adjustments")
    if gpu_clamps_total <= 0:
        raise RuntimeError("pressure gate: expected gpu clamp events")
    if final_inflight > initial_inflight:
        raise RuntimeError(
            "pressure gate: expected inflight cap to stay same or decrease under high pressure"
        )
    if final_inflight > min_expected:
        raise RuntimeError(
            f"pressure gate: expected hard clamp <= min expected ({min_expected}), got {final_inflight}"
        )

    print("video_nvdec_pressure_gate_summary")
    print(f"  batches: {batches}")
    print(f"  samples: {samples}")
    print(f"  initial_max_inflight_bytes: {initial_inflight}")
    print(f"  final_max_inflight_bytes: {final_inflight}")
    print(f"  min_expected_inflight_bytes: {min_expected}")
    print(f"  pressure: {pressure:.3f}")
    print(f"  gpu_pressure: {gpu_pressure:.3f}")
    print(f"  autotune_adjustments_total: {adjustments_total}")
    print(f"  gpu_clamps_total: {gpu_clamps_total}")


if __name__ == "__main__":
    main()
