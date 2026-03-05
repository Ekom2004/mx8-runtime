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


def _make_mp4(path: Path, color: str, *, size: str, fps: int = 8, seconds: int = 2) -> None:
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
        f"color=c={color}:s={size}:r={fps}:d={seconds}",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed creating {path}: {proc.stderr.strip() or proc.stdout.strip()}"
        )


def _prepare_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _make_mp4(root / "1080p_red.mp4", "red", size="1920x1080")
    _make_mp4(root / "1080p_green.mp4", "green", size="1920x1080")
    _make_mp4(root / "4k_blue.mp4", "blue", size="3840x2160")
    _make_mp4(root / "4k_yellow.mp4", "yellow", size="3840x2160")


def _run_backend(backend: str, data_root: Path, store_root: Path) -> dict:
    os.environ["MX8_VIDEO_DECODE_BACKEND"] = backend
    loader = mx8.video(
        str(data_root),
        store=str(store_root),
        recursive=True,
        clip=4,
        stride=2,
        fps=8,
        batch=2,
        ram_gb=8,
        profile="throughput",
        tune=False,
    )
    started = time.perf_counter()
    batches = 0
    samples = 0
    for _idx, batch in zip(range(16), loader):
        batches += 1
        samples += len(batch.sample_ids)
        if samples >= 8:
            break
    elapsed = max(1e-6, time.perf_counter() - started)
    stats = dict(loader.stats())
    return {
        "backend": backend,
        "batches": batches,
        "samples": samples,
        "elapsed_s": elapsed,
        "samples_per_s": samples / elapsed,
        "fallback_total": int(stats.get("video_decode_backend_fallback_total", 0)),
        "decode_ms_total": int(stats.get("video_decode_ms_total", 0)),
        "stats": stats,
    }


def main() -> None:
    tmp_root = Path(os.environ["MX8_VIDEO_NVDEC_THROUGHPUT_TMP_ROOT"])
    store_root = tmp_root / "store"
    data_root = tmp_root / "videos"
    store_root.mkdir(parents=True, exist_ok=True)
    _prepare_dataset(data_root)

    os.environ["MX8_VIDEO_STAGE1_INDEX"] = "1"
    os.environ["MX8_VIDEO_STAGE2_BYTES_PER_CLIP"] = os.environ.get(
        "MX8_VIDEO_STAGE2_BYTES_PER_CLIP", "4096"
    )
    os.environ.pop("MX8_VIDEO_GPU_PRESSURE_RATIO", None)

    cli = _run_backend("cli", data_root, store_root)
    nvdec = _run_backend("nvdec", data_root, store_root)

    if cli["samples"] <= 0 or nvdec["samples"] <= 0:
        raise RuntimeError("throughput gate: expected both backends to deliver samples")

    require_hw = os.environ.get("MX8_VIDEO_NVDEC_THROUGHPUT_REQUIRE_HW", "0") in (
        "1",
        "true",
        "yes",
    )
    min_speedup = float(os.environ.get("MX8_VIDEO_NVDEC_MIN_SPEEDUP", "1.05"))
    nvdec_has_hw_path = nvdec["fallback_total"] == 0
    if require_hw and not nvdec_has_hw_path:
        raise RuntimeError(
            "throughput gate: hardware NVDEC path required but backend fell back"
        )
    if nvdec_has_hw_path and nvdec["samples_per_s"] < cli["samples_per_s"] * min_speedup:
        raise RuntimeError(
            "throughput gate: expected nvdec throughput improvement "
            f"(need >= {min_speedup:.2f}x, got {nvdec['samples_per_s'] / max(1e-6, cli['samples_per_s']):.2f}x)"
        )

    print("video_nvdec_throughput_gate_summary")
    print(f"  cli_samples_per_s: {cli['samples_per_s']:.2f}")
    print(f"  nvdec_samples_per_s: {nvdec['samples_per_s']:.2f}")
    print(f"  cli_decode_ms_total: {cli['decode_ms_total']}")
    print(f"  nvdec_decode_ms_total: {nvdec['decode_ms_total']}")
    print(f"  nvdec_fallback_total: {nvdec['fallback_total']}")
    if nvdec_has_hw_path:
        print(
            f"  nvdec_speedup_vs_cli: {nvdec['samples_per_s'] / max(1e-6, cli['samples_per_s']):.2f}x"
        )
    else:
        print("  nvdec_speedup_vs_cli: skipped (fallback path on this host)")


if __name__ == "__main__":
    main()
